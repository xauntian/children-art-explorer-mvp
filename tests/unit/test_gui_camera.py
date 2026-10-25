"""GUI 相机规则测试：所有 ADB 均模拟，禁止真实拍照和修改手机设置。"""

from __future__ import annotations

import io
import subprocess

import pytest
from PIL import Image

from app.errors import CaptureError
from app.gui_camera import GuiCamera
from app.phone_capture import PhoneCaptureResult


def png(width=200, height=400, color=(230, 180, 70)):
    output = io.BytesIO()
    Image.new("RGB", (width, height), color).save(output, format="PNG")
    return output.getvalue()


class FakeAdb:
    def __init__(self):
        self.screenshot = png()
        self.devices = b"List of devices attached\nPHONE\tdevice model:Test\n"
        self.policy = "mIsShowing=false"
        self.foreground = "com.huawei.camera"
        self.calls = []

    def __call__(self, command, **kwargs):
        args = command[1:]
        if args[:1] == ["-s"]:
            assert args[1] == "PHONE"
            args = args[2:]
        self.calls.append(args)
        assert kwargs["text"] is False
        assert kwargs["timeout"] > 0
        if args == ["devices", "-l"]:
            output = self.devices
        elif args == ["shell", "dumpsys", "window", "policy"]:
            output = self.policy.encode()
        elif args == ["shell", "dumpsys", "activity", "activities"]:
            output = f"mResumedActivity: ActivityRecord{{1 u0 {self.foreground}/.Camera}}".encode()
        elif args == ["exec-out", "screencap", "-p"]:
            output = self.screenshot
        else:
            raise AssertionError(f"禁止的 ADB 命令: {args}")
        return subprocess.CompletedProcess(command, 0, output, b"")


@pytest.fixture
def camera_env(tmp_path, monkeypatch):
    adb = tmp_path / "tools" / "android" / "platform-tools" / "adb.exe"
    adb.parent.mkdir(parents=True)
    adb.touch()
    fake = FakeAdb()
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    config = {
        "serial": "PHONE",
        "preview_screen_size": [200, 400],
        "preview_roi": [0.1, 0.2, 0.9, 0.8],
        # 即使旧配置允许点击唤醒与快门，GUI 也必须完全忽略。
        "fallback_photo": True,
        "camera_shutter": {
            "method": "tap",
            "camera_package": "com.huawei.camera",
            "screen_size": [200, 400],
            "position": [100, 380],
            "wake_position": [100, 200],
        },
    }
    return GuiCamera(tmp_path, config), fake, tmp_path


def test_capture_preserves_original_png_and_roi_without_camera_writes(camera_env):
    camera, fake, _ = camera_env
    result = camera.capture()
    assert isinstance(result, PhoneCaptureResult)
    assert result.raw_bytes == fake.screenshot
    assert result.source == "preview_screenshot"
    assert result.preview_image.shape == (240, 160, 3)
    assert result.meta["preview_bounds_px"] == [20, 80, 180, 320]
    assert result.meta["shutter_count"] == result.meta["wake_count"] == 0
    assert result.meta["camera_package"] == "com.huawei.camera"
    assert not any("zoom" in key for key in result.meta)
    assert camera.phone.fallback_photo is False
    assert camera.phone.camera_shutter["wake_position"] is None
    assert fake.calls == [
        ["devices", "-l"],
        ["shell", "dumpsys", "window", "policy"],
        ["shell", "dumpsys", "activity", "activities"],
        ["exec-out", "screencap", "-p"],
    ]


def test_prepare_reports_only_device_availability(camera_env):
    camera, fake, _ = camera_env
    assert camera.prepare() == {
        "available": True,
        "serial": "PHONE",
        "camera_package": "com.huawei.camera",
        "screen_size": [200, 400],
        "error": None,
    }
    assert len([call for call in fake.calls if "screencap" in call]) == 1


def test_prepare_report_cannot_change_stored_report(camera_env):
    camera, _, _ = camera_env
    report = camera.prepare()
    report["available"] = False
    assert camera._last_report["available"] is True


def test_probe_keeps_same_read_only_checks(camera_env):
    camera, fake, _ = camera_env
    assert camera.probe()["available"] is True
    assert len(fake.calls) == 4


def test_each_capture_reads_a_new_screenshot_without_cached_pixels(camera_env):
    camera, fake, _ = camera_env
    first = camera.capture()
    fake.calls.clear()
    fake.screenshot = png(color=(100, 200, 220))
    second = camera.capture()
    assert second.raw_bytes == fake.screenshot
    assert second.raw_bytes != first.raw_bytes
    assert second.preview_image[0, 0].tolist() == [220, 200, 100]
    assert len(fake.calls) == 4
    assert len([call for call in fake.calls if "screencap" in call]) == 1


@pytest.mark.parametrize("entry", ["capture", "prepare", "probe"])
def test_locked_phone_stops_before_screenshot_without_waking(camera_env, entry):
    camera, fake, _ = camera_env
    fake.policy = "mIsShowing=true"
    if entry == "capture":
        with pytest.raises(CaptureError, match="手机已锁屏"):
            camera.capture()
    else:
        report = getattr(camera, entry)()
        assert report["available"] is False
        assert "手机已锁屏" in report["error"]
    assert fake.calls == [["devices", "-l"], ["shell", "dumpsys", "window", "policy"]]


@pytest.mark.parametrize("entry", ["capture", "prepare", "probe"])
def test_other_foreground_never_gets_any_screenshot_or_camera_action(camera_env, entry):
    camera, fake, _ = camera_env
    fake.foreground = "com.example.other"
    if entry == "capture":
        with pytest.raises(CaptureError, match="前台不是已配置相机"):
            camera.capture()
    else:
        report = getattr(camera, entry)()
        assert report["available"] is False
        assert "前台不是已配置相机" in report["error"]
    assert not any("screencap" in call or "input" in call for call in fake.calls)


def test_unknown_lock_state_stops_before_foreground_and_screenshot(camera_env):
    camera, fake, _ = camera_env
    fake.policy = "unsupported policy"
    report = camera.prepare()
    assert report["available"] is False
    assert "无法确认手机当前锁屏状态" in report["error"]
    assert len(fake.calls) == 2


def test_black_screen_never_wakes_or_falls_back_even_with_legacy_config(camera_env):
    camera, fake, _ = camera_env
    fake.screenshot = png(color=(0, 0, 0))
    with pytest.raises(CaptureError, match="黑屏"):
        camera.capture(preview_only=False)
    assert not any("input" in call for call in fake.calls)
    assert len([call for call in fake.calls if "screencap" in call]) == 1


def test_rotated_or_changed_screen_fails_without_using_stale_roi(camera_env):
    camera, fake, _ = camera_env
    fake.screenshot = png(400, 200)
    report = camera.prepare()
    assert report["available"] is False
    assert "标定尺寸" in report["error"]
    assert not any("input" in call for call in fake.calls)


def test_expected_same_phone_is_not_silently_replaced(camera_env):
    camera, fake, _ = camera_env
    camera.prepare()
    fake.devices = b"List of devices attached\nSECOND\tdevice model:Test\n"
    report = camera.prepare()
    assert report["available"] is False
    assert report["serial"] == "PHONE"
    assert "未找到指定手机" in report["error"]


def test_unconfigured_single_device_is_pinned_for_later_capture(camera_env):
    _, fake, root = camera_env
    camera = GuiCamera(root)
    assert camera.prepare()["serial"] == "PHONE"
    assert camera.phone.serial == "PHONE"
    fake.devices = b"List of devices attached\nSECOND\tdevice model:Test\n"
    assert camera.prepare()["available"] is False


def test_first_selected_device_stays_pinned_after_foreground_failure(camera_env):
    _, fake, root = camera_env
    camera = GuiCamera(root)
    fake.foreground = "com.example.other"
    assert camera.prepare()["available"] is False
    assert camera.phone.serial == "PHONE"
    fake.devices = b"List of devices attached\nSECOND\tdevice model:Test\n"
    assert "未找到指定手机" in camera.prepare()["error"]


@pytest.mark.parametrize("state", ["unauthorized", "offline"])
def test_unavailable_device_stops_before_accessing_screen(camera_env, state):
    camera, fake, _ = camera_env
    fake.devices = f"List of devices attached\nPHONE\t{state}\n".encode()
    report = camera.prepare()
    assert report["available"] is False
    assert "未授权或不可用" in report["error"]
    assert fake.calls == [["devices", "-l"]]


def test_unconfigured_multiple_devices_are_not_arbitrarily_selected(camera_env):
    _, fake, root = camera_env
    camera = GuiCamera(root)
    fake.devices = b"List of devices attached\nPHONE\tdevice\nSECOND\tdevice\n"
    report = camera.prepare()
    assert report["available"] is False
    assert "多个已授权手机" in report["error"]
    assert fake.calls == [["devices", "-l"]]


@pytest.mark.parametrize("raw", [b"", b"not a PNG", b"\x89PNG\r\n\x1a\ninvalid"])
def test_incomplete_or_non_png_capture_is_rejected_without_fallback(camera_env, raw):
    camera, fake, _ = camera_env
    fake.screenshot = raw
    with pytest.raises(CaptureError):
        camera.capture()
    assert len(fake.calls) == 4
    assert not any("input" in call for call in fake.calls)


@pytest.mark.parametrize("args", [
    ["shell", "input", "tap", "100", "200"],
    ["shell", "input", "keyevent", "KEYCODE_CAMERA"],
    ["shell", "am", "start", "com.huawei.camera/.Camera"],
    ["shell", "settings", "put", "system", "screen_off_timeout", "600000"],
    ["kill-server"],
    ["push", "server", "/data/local/tmp/server"],
    ["exec-out", "uiautomator", "dump", "/dev/tty"],
])
def test_allowlist_blocks_camera_mutation_and_ui_dump_before_subprocess(camera_env, args):
    camera, fake, _ = camera_env
    with pytest.raises(CaptureError, match="只允许读取"):
        camera.phone._run(args, serial="PHONE", timeout=1)
    assert fake.calls == []


def test_failed_prepare_replaces_previous_success_report(camera_env):
    camera, fake, _ = camera_env
    assert camera.prepare()["available"] is True
    fake.policy = "mIsShowing=true"
    report = camera.prepare()
    assert report["available"] is False
    assert report["screen_size"] is None
    assert "手机已锁屏" in report["error"]


def test_close_only_releases_local_state_and_rejects_further_capture(camera_env):
    camera, fake, _ = camera_env
    camera.prepare()
    fake.calls.clear()
    camera.close()
    assert fake.calls == []
    with pytest.raises(CaptureError, match="已关闭"):
        camera.capture()
    assert camera.prepare()["available"] is False
    assert camera.probe()["available"] is False
    assert fake.calls == []
