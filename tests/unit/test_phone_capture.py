"""手机采集规则测试：所有 subprocess 均模拟，不连接手机、不拍照。"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.errors import CaptureError
from app.phone_capture import PhoneCapture


def image_bytes(width=160, height=100, color=(230, 180, 70), *, image_format="PNG", exif=None):
    output = io.BytesIO()
    options = {"exif": exif} if exif else {}
    Image.new("RGB", (width, height), color).save(output, format=image_format, **options)
    return output.getvalue()


class Clock:
    def __init__(self):
        self.value = 100.0

    def now(self):
        return self.value

    def sleep(self, duration):
        self.value += max(duration, 0.001)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    adb = tmp_path / "tools" / "android" / "platform-tools" / "adb.exe"
    adb.parent.mkdir(parents=True)
    adb.touch()
    clock = Clock()
    monkeypatch.setattr("app.phone_capture.time.monotonic", clock.now)
    monkeypatch.setattr("app.phone_capture.time.sleep", clock.sleep)
    return tmp_path, adb, clock


class FakeAdb:
    def __init__(self, screenshot=None, devices=None):
        self.screenshot = screenshot if screenshot is not None else image_bytes()
        self.devices = (
            devices
            if devices is not None
            else b"List of devices attached\nPHONE\tdevice model:Test\n"
        )
        self.calls = []
        self.baseline = {"/sdcard/DCIM/Camera/old.jpg": image_bytes(image_format="JPEG")}
        self.photos = dict(self.baseline)
        self.shutters = 0
        self.on_shutter = None
        self.on_list = None
        self.list_count = 0
        self.sizes = None
        self.foreground = "com.huawei.camera"
        self.keyguard_policy = (
            "deviceHasKeyguard=true\nsecure=true\nshowing=false\nmIsShowing=false"
        )
        self.tap_positions = []
        self.shutter_position = None
        self.wake_position = None
        self.on_wake = None

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        assert kwargs["text"] is False
        assert kwargs["timeout"] > 0
        assert kwargs["capture_output"] is True
        args = command[1:]
        if args[:1] == ["-s"]:
            assert args[1] in {"PHONE", "SECOND"}
            args = args[2:]
        if args == ["devices", "-l"]:
            return subprocess.CompletedProcess(command, 0, self.devices, b"")
        if args == ["exec-out", "screencap", "-p"]:
            if isinstance(self.screenshot, Exception):
                raise self.screenshot
            return subprocess.CompletedProcess(command, 0, self.screenshot, b"")
        if args == ["shell", "dumpsys", "activity", "activities"]:
            output = f"mResumedActivity: ActivityRecord{{1 u0 {self.foreground}/.Camera}}"
            return subprocess.CompletedProcess(command, 0, output.encode(), b"")
        if args == ["shell", "dumpsys", "window", "policy"]:
            return subprocess.CompletedProcess(command, 0, self.keyguard_policy.encode(), b"")
        if args[:3] == ["shell", "input", "tap"]:
            position = [int(args[3]), int(args[4])]
            self.tap_positions.append(position)
            if position == self.wake_position:
                if self.on_wake:
                    self.on_wake(self)
            elif position == self.shutter_position:
                self.shutters += 1
                if self.on_shutter:
                    self.on_shutter(self)
            else:
                raise AssertionError(f"未配置的触控坐标: {position}")
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if args == ["shell", "input", "keyevent", "KEYCODE_CAMERA"]:
            self.shutters += 1
            if self.on_shutter:
                self.on_shutter(self)
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if len(args) == 2 and args[0] == "shell" and args[1].startswith("find "):
            self.list_count += 1
            if self.on_list:
                self.on_list(self)
            data = b"".join(path.encode() + b"\x00" for path in self.photos)
            return subprocess.CompletedProcess(command, 0, data, b"")
        if len(args) == 2 and args[0] == "shell" and args[1].startswith("stat "):
            import shlex

            path = shlex.split(args[1])[-1]
            size = self.sizes.pop(0) if self.sizes else len(self.photos[path])
            return subprocess.CompletedProcess(command, 0, str(size).encode(), b"")
        if args[:1] == ["pull"]:
            Path(args[2]).write_bytes(self.photos[args[1]])
            return subprocess.CompletedProcess(command, 0, b"1 file pulled", b"")
        raise AssertionError(f"未预期的 ADB 命令: {args}")


def test_screenshot_keeps_original_png_and_only_crops_preview(environment, monkeypatch):
    root, adb, _ = environment
    raw = image_bytes(200, 100)
    fake = FakeAdb(raw)
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(root, preview_roi=[0.1, 0.2, 0.9, 0.8]).capture()
    assert result.raw_bytes == raw
    assert result.source == "preview_screenshot"
    assert result.meta["source"] == "preview_screenshot"
    assert result.meta["raw_format"] == "png"
    assert result.meta["raw_suffix"] == ".png"
    assert result.preview_image.shape == (60, 160, 3)
    assert result.preview_image[0, 0].tolist() == [70, 180, 230]
    assert result.meta["orientation"] == "landscape"
    assert result.meta["preview_bounds_px"] == [20, 20, 180, 80]
    assert fake.shutters == 0
    assert all(Path(call[0][0]) == adb for call in fake.calls)
    assert fake.calls[2][0][1:3] == ["-s", "PHONE"]
    assert fake.calls[2][1]["timeout"] == 3.0


def test_none_roi_preserves_full_screenshot_and_portrait(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(image_bytes(100, 180))
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(root).capture(preview_only=True)
    assert result.preview_image.shape == (180, 100, 3)
    assert result.meta["orientation"] == "portrait"
    assert result.meta["preview_roi"] is None


@pytest.mark.parametrize(
    "devices", [b"List of devices attached\n", b"PHONE\tunauthorized\n", b"PHONE\toffline\n"]
)
def test_requires_authorized_device_without_shutter(environment, monkeypatch, devices):
    root, _, _ = environment
    fake = FakeAdb(devices=devices)
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="没有已授权"):
        PhoneCapture(root).capture()
    assert len(fake.calls) == 1
    assert fake.shutters == 0


def test_multiple_devices_require_explicit_serial(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(devices=b"PHONE\tdevice\nSECOND\tdevice\n")
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="多个"):
        PhoneCapture(root).capture()
    result = PhoneCapture(root, serial="SECOND").capture()
    assert result.meta["serial"] == "SECOND"


@pytest.mark.parametrize("serial, expected", [("MISSING", "未找到"), ("PHONE", "未授权")])
def test_configured_serial_must_be_present_and_authorized(
    environment, monkeypatch, serial, expected
):
    root, _, _ = environment
    fake = FakeAdb(devices=b"PHONE\tunauthorized\nSECOND\tdevice\n")
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match=expected):
        PhoneCapture(root, serial=serial).capture()
    assert fake.shutters == 0


def test_missing_project_adb_never_uses_path(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("不得寻找系统 adb")

    monkeypatch.setattr("app.phone_capture.subprocess.run", unexpected)
    with pytest.raises(CaptureError, match="项目 ADB 不存在"):
        PhoneCapture(tmp_path).capture()


@pytest.mark.parametrize("preview_only, fallback_photo", [(True, True), (False, False)])
def test_bad_preview_does_not_shoot_when_forbidden(
    environment, monkeypatch, preview_only, fallback_photo
):
    root, _, _ = environment
    fake = FakeAdb(b"not png")
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="PNG"):
        PhoneCapture(root, fallback_photo=fallback_photo).capture(preview_only=preview_only)
    assert fake.shutters == 0
    assert len(fake.calls) == 3


def test_black_preview_probe_is_read_only(environment, monkeypatch):
    root, _, _ = environment
    screen = np.zeros((200, 100, 3), dtype=np.uint8)
    screen[-20:, :] = 255  # 黑屏上的快门控件不能掩盖黑预览。
    output = io.BytesIO()
    Image.fromarray(screen).save(output, format="PNG")
    fake = FakeAdb(output.getvalue())
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    report = PhoneCapture(root).probe()
    assert report["available"] is False
    assert "黑屏" in report["error"]
    assert fake.shutters == 0


@pytest.mark.parametrize(
    "failure",
    [b"bad screenshot", image_bytes(color=(0, 0, 0)), subprocess.TimeoutExpired("adb", 3)],
)
def test_preview_failure_falls_back_once_to_new_stable_photo(environment, monkeypatch, failure):
    root, _, _ = environment
    fake = FakeAdb(failure)
    new_path = "/sdcard/DCIM/Camera/new artwork.jpg"
    raw = image_bytes(240, 160, image_format="JPEG")
    fake.on_shutter = lambda owner: owner.photos.update({new_path: raw})
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(root, preview_roi=[0.2, 0.2, 0.8, 0.8]).capture()
    assert result.source == "camera_photo"
    assert result.meta["source"] == "camera_photo"
    assert result.meta["raw_format"] == "jpeg"
    assert result.meta["raw_suffix"] == ".jpg"
    assert result.raw_bytes == raw
    assert result.preview_image.shape == (160, 240, 3)
    assert result.meta["remote_photo"] == new_path
    assert result.meta["remote_photo_preserved"] is True
    assert result.meta["shutter_count"] == fake.shutters == 1
    assert result.meta["preview_roi"] is None
    assert fake.photos[new_path] == raw
    assert all("rm" not in call[0] for call in fake.calls)


def test_fallback_never_reuses_existing_photo(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(b"bad screenshot")
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="未复用旧照片"):
        PhoneCapture(root, photo_timeout_sec=0.6).capture()
    assert fake.shutters == 1
    assert not any("pull" in call[0] for call in fake.calls)


def test_fallback_waits_for_positive_stable_size(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(b"bad screenshot")
    raw = image_bytes(image_format="JPEG")
    fake.on_shutter = lambda owner: owner.photos.update({"/sdcard/DCIM/Camera/new.jpg": raw})
    fake.sizes = [0, 12, len(raw), len(raw), len(raw)]
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(root).capture()
    assert result.raw_bytes == raw
    assert len([call for call in fake.calls if "pull" in call[0]]) == 1


def test_corrupt_new_photo_is_rejected_without_second_shutter(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(b"bad screenshot")
    fake.on_shutter = lambda owner: owner.photos.update(
        {"/sdcard/DCIM/Camera/new.jpg": b"not an image"}
    )
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="无法完整解码"):
        PhoneCapture(root, photo_timeout_sec=0.6).capture()
    assert fake.shutters == 1
    assert len([call for call in fake.calls if "pull" in call[0]]) == 1


def test_multiple_new_photos_are_ambiguous(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(b"bad screenshot")
    fake.on_shutter = lambda owner: owner.photos.update(
        {
            "/sdcard/DCIM/Camera/new1.jpg": image_bytes(),
            "/sdcard/DCIM/Camera/new2.jpg": image_bytes(),
        }
    )
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="多张新照片"):
        PhoneCapture(root).capture()
    assert fake.shutters == 1
    assert not any("pull" in call[0] for call in fake.calls)


def test_native_photo_exif_rotation_applies_only_to_decoded_image(environment, monkeypatch):
    root, _, _ = environment
    exif = Image.Exif()
    exif[274] = 6
    raw = image_bytes(160, 100, image_format="JPEG", exif=exif)
    fake = FakeAdb(b"bad screenshot")
    fake.on_shutter = lambda owner: owner.photos.update({"/sdcard/DCIM/Camera/new.jpg": raw})
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(root).capture()
    assert result.raw_bytes == raw
    assert result.preview_image.shape[:2] == (160, 100)
    assert result.meta["exif_orientation"] == 6
    assert result.meta["raw_width"] == 160
    assert result.meta["orientation"] == "portrait"


def test_every_subprocess_is_timeout_bounded_in_fallback(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(b"bad screenshot")
    fake.on_shutter = lambda owner: owner.photos.update(
        {"/sdcard/DCIM/Camera/new.jpg": image_bytes()}
    )
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    PhoneCapture(root, command_timeout_sec=1.0, photo_timeout_sec=0.8).capture()
    assert all(0 < kwargs["timeout"] <= 1 for _, kwargs in fake.calls if kwargs["timeout"] != 3)
    assert all(kwargs["timeout"] <= 0.8 for command, kwargs in fake.calls[3:])


def test_transport_failure_is_capture_error(environment, monkeypatch):
    root, _, _ = environment

    def fail(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, b"", b"device offline")

    monkeypatch.setattr("app.phone_capture.subprocess.run", fail)
    with pytest.raises(CaptureError, match="device offline"):
        PhoneCapture(root).capture()


def test_truncated_png_cannot_pass_as_preview(environment, monkeypatch):
    root, _, _ = environment
    raw = image_bytes()
    fake = FakeAdb(raw[: len(raw) // 2])
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="无法完整解码"):
        PhoneCapture(root).capture(preview_only=True)
    assert fake.shutters == 0


def test_screenshot_deadline_includes_decoding(environment, monkeypatch):
    root, _, clock = environment
    fake = FakeAdb()
    decode = PhoneCapture._decode

    def slow_decode(data, **kwargs):
        decoded = decode(data, **kwargs)
        clock.value += 3.1
        return decoded

    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    monkeypatch.setattr(PhoneCapture, "_decode", staticmethod(slow_decode))
    with pytest.raises(CaptureError, match="截止时间"):
        PhoneCapture(root).capture(preview_only=True)
    assert fake.shutters == 0


def test_pull_timeout_never_triggers_a_second_shutter(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(b"bad screenshot")
    fake.on_shutter = lambda owner: owner.photos.update(
        {"/sdcard/DCIM/Camera/new.jpg": image_bytes()}
    )

    def run(command, **kwargs):
        if "pull" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return fake(command, **kwargs)

    monkeypatch.setattr("app.phone_capture.subprocess.run", run)
    with pytest.raises(CaptureError, match="ADB 命令超时"):
        PhoneCapture(root).capture()
    assert fake.shutters == 1


def test_photo_size_changing_during_pull_is_not_accepted(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(b"bad screenshot")
    raw = image_bytes()
    fake.on_shutter = lambda owner: owner.photos.update({"/sdcard/DCIM/Camera/new.jpg": raw})
    fake.sizes = [len(raw), len(raw), len(raw) + 1, len(raw), len(raw), len(raw)]
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(root).capture()
    assert result.raw_bytes == raw
    assert fake.shutters == 1
    assert len([call for call in fake.calls if "pull" in call[0]]) == 2


@pytest.mark.parametrize(
    "roi", [[0, 0, 0, 1], [0, 0, 2, 1], [0, 0, 1], [0, float("nan"), 1, 1], [False, 0, 1, 1]]
)
def test_invalid_roi_rejected_before_device_access(tmp_path, roi):
    with pytest.raises(CaptureError, match="preview_roi"):
        PhoneCapture(tmp_path, preview_roi=roi)


def test_calibrated_screen_size_applies_roi_to_the_intended_pixels(environment, monkeypatch):
    root, _, _ = environment
    raw = image_bytes(1344, 2772)
    fake = FakeAdb(raw)
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(
        root,
        preview_screen_size=[1344, 2772],
        preview_roi=[28 / 1344, 350 / 2772, 1316 / 1344, 2082 / 2772],
    ).capture(preview_only=True)
    assert result.meta["preview_screen_size"] == [1344, 2772]
    assert result.meta["preview_bounds_px"] == [28, 350, 1316, 2082]
    assert result.preview_image.shape == (1732, 1288, 3)
    assert result.raw_bytes == raw
    assert result.source == "preview_screenshot"
    assert fake.shutters == 0


@pytest.mark.parametrize("preview_only, fallback_photo", [(True, True), (False, False)])
def test_rotated_screen_rejects_stale_roi_without_shutter_when_forbidden(
    environment, monkeypatch, preview_only, fallback_photo
):
    root, _, _ = environment
    fake = FakeAdb(image_bytes(200, 100))
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    phone = PhoneCapture(
        root,
        preview_screen_size=[100, 200],
        preview_roi=[0.1, 0.2, 0.9, 0.8],
        fallback_photo=fallback_photo,
    )
    with pytest.raises(CaptureError, match="与预览标定尺寸.*不一致") as error:
        phone.capture(preview_only=preview_only)
    assert error.value.details == {
        "actual_screen_size": [200, 100],
        "expected_screen_size": [100, 200],
    }
    assert fake.shutters == 0
    assert len(fake.calls) == 3


def test_screen_size_mismatch_probe_is_read_only(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(image_bytes(200, 100))
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    report = PhoneCapture(root, preview_screen_size=[100, 200]).probe()
    assert report["available"] is False
    assert "不一致" in report["error"]
    assert fake.shutters == 0


def test_screen_size_mismatch_can_fall_back_to_one_full_native_photo(environment, monkeypatch):
    root, _, _ = environment
    fake = FakeAdb(image_bytes(200, 100))
    native = image_bytes(320, 180, image_format="JPEG")
    fake.on_shutter = lambda owner: owner.photos.update({"/sdcard/DCIM/Camera/new.jpg": native})
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(
        root,
        preview_screen_size=[100, 200],
        preview_roi=[0.1, 0.2, 0.9, 0.8],
    ).capture()
    assert result.source == "camera_photo"
    assert result.raw_bytes == native
    assert result.meta["preview_screen_size"] is None
    assert result.meta["preview_roi"] is None
    assert result.preview_image.shape == (180, 320, 3)
    assert "不一致" in result.meta["fallback_reason"]
    assert fake.shutters == 1


@pytest.mark.parametrize(
    "screen_size",
    [
        [],
        [1344],
        [1344, 2772, 1],
        [1344, 0],
        [-1, 2772],
        [True, 2772],
        [1344.0, 2772],
        ["1344", 2772],
    ],
)
def test_invalid_screen_size_rejected_before_device_access(tmp_path, screen_size):
    with pytest.raises(CaptureError, match="preview_screen_size"):
        PhoneCapture(tmp_path, preview_screen_size=screen_size)


def test_explicit_photo_fallback_skips_screenshot_and_only_triggers_one_shutter(
    environment, monkeypatch
):
    root, _, _ = environment
    fake = FakeAdb()
    photo = image_bytes(320, 180, image_format="JPEG")
    fake.on_shutter = lambda owner: owner.photos.update({"/sdcard/DCIM/Camera/new.jpg": photo})
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    result = PhoneCapture(root).capture_photo(reason="camera UI occludes artwork")
    assert result.source == "camera_photo"
    assert result.raw_bytes == photo
    assert result.meta["fallback_reason"] == "camera UI occludes artwork"
    assert result.meta["shutter_count"] == fake.shutters == 1
    assert result.meta["remote_photo_preserved"] is True
    assert not any("screencap" in command for command, _ in fake.calls)


def test_explicit_photo_fallback_respects_disabled_photo_policy(environment, monkeypatch):
    root, _, _ = environment

    def unexpected(*args, **kwargs):
        raise AssertionError("禁用回退时不访问设备")

    monkeypatch.setattr("app.phone_capture.subprocess.run", unexpected)
    with pytest.raises(CaptureError, match="已禁用"):
        PhoneCapture(root, fallback_photo=False).capture_photo(reason="camera UI occludes artwork")


@pytest.mark.parametrize("reason", ["", "  ", None])
def test_explicit_photo_fallback_requires_reason_before_device_access(
    environment, monkeypatch, reason
):
    root, _, _ = environment

    def unexpected(*args, **kwargs):
        raise AssertionError("没有回退原因时不访问设备")

    monkeypatch.setattr("app.phone_capture.subprocess.run", unexpected)
    with pytest.raises(CaptureError, match="reason"):
        PhoneCapture(root).capture_photo(reason=reason)


def _tap_config():
    return {
        "method": "tap",
        "camera_package": "com.huawei.camera",
        "screen_size": [200, 100],
        "position": [100, 90],
        "wake_position": [100, 50],
        "wake_delay_sec": 0.1,
    }


def _tap_fake(monkeypatch, *, asleep=False):
    fake = FakeAdb(image_bytes(200, 100, color=(0, 0, 0) if asleep else (220, 180, 100)))
    fake.shutter_position = [100, 90]
    fake.wake_position = [100, 50]
    fake.on_wake = lambda owner: setattr(owner, "screenshot", image_bytes(200, 100))
    photo = image_bytes(320, 180, image_format="JPEG")
    fake.on_shutter = lambda owner: owner.photos.update({"/sdcard/DCIM/Camera/new.jpg": photo})
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    return fake


def test_calibrated_tap_shutter_takes_one_new_photo_without_keyevent(environment, monkeypatch):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch)
    result = PhoneCapture(root, camera_shutter=_tap_config()).capture_photo(reason="UI occlusion")
    assert result.source == "camera_photo"
    assert result.meta["shutter_method"] == "tap"
    assert result.meta["camera_package"] == "com.huawei.camera"
    assert result.meta["wake_count"] == 0
    assert result.meta["shutter_count"] == fake.shutters == 1
    assert fake.tap_positions == [[100, 90]]
    assert not any("keyevent" in command for command, _ in fake.calls)


def test_black_camera_is_woken_once_before_native_shutter(environment, monkeypatch):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    result = PhoneCapture(root, camera_shutter=_tap_config()).capture_photo(reason="UI occlusion")
    assert result.meta["wake_count"] == 1
    assert fake.shutters == 1
    assert fake.tap_positions == [[100, 50], [100, 90]]


def test_main_preview_wakes_camera_and_returns_fresh_png_without_photograph(
    environment, monkeypatch
):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    old_png = fake.screenshot
    result = PhoneCapture(root, camera_shutter=_tap_config()).capture()
    assert result.source == "preview_screenshot"
    assert result.raw_bytes != old_png
    assert result.raw_bytes == fake.screenshot
    assert result.meta["wake_count"] == 1
    assert fake.shutters == 0
    assert fake.tap_positions == [[100, 50]]


@pytest.mark.parametrize("entry", ["probe", "preview_only"])
def test_read_only_preview_never_wakes_camera(environment, monkeypatch, entry):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    phone = PhoneCapture(root, camera_shutter=_tap_config())
    if entry == "probe":
        assert phone.probe()["available"] is False
    else:
        with pytest.raises(CaptureError, match="黑屏"):
            phone.capture(preview_only=True)
    assert fake.tap_positions == []
    assert fake.shutters == 0


def test_wrong_foreground_never_receives_wake_or_shutter(environment, monkeypatch):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    fake.foreground = "com.example.other"
    with pytest.raises(CaptureError, match="前台不是已配置相机"):
        PhoneCapture(root, camera_shutter=_tap_config()).capture_photo(reason="UI occlusion")
    assert fake.tap_positions == []
    assert fake.shutters == 0


def test_tap_shutter_rejects_changed_screen_size_before_any_touch(environment, monkeypatch):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch)
    fake.screenshot = image_bytes(100, 200)
    with pytest.raises(CaptureError, match="触控标定尺寸.*不一致"):
        PhoneCapture(root, camera_shutter=_tap_config()).capture_photo(reason="UI occlusion")
    assert fake.tap_positions == []
    assert fake.shutters == 0


def test_wake_without_ready_preview_does_not_repeat_wake_or_fire_shutter(environment, monkeypatch):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    fake.on_wake = None
    with pytest.raises(CaptureError, match="黑屏"):
        PhoneCapture(root, camera_shutter=_tap_config()).capture()
    assert fake.tap_positions == [[100, 50]]
    assert fake.shutters == 0


def test_keyevent_default_checks_camera_foreground_without_touching_other_app(
    environment, monkeypatch
):
    root, _, _ = environment
    fake = FakeAdb()
    fake.foreground = "com.example.other"
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    with pytest.raises(CaptureError, match="前台不是已配置相机"):
        PhoneCapture(root).capture_photo(reason="preview failed")
    assert fake.shutters == 0
    assert fake.tap_positions == []


def test_grey_idle_instruction_does_not_hide_black_preview():
    screen = np.zeros((1000, 600, 3), dtype=np.uint8)
    # 模拟“点击屏幕唤醒相机”稀疏灰字，不将待机屏误判为可用画面。
    screen[490:500, 200:400:4] = 130
    assert PhoneCapture._black_stats(screen)["black"] is True


@pytest.mark.parametrize(
    "shutter",
    [
        "tap",
        {"method": "unknown"},
        {"method": "tap"},
        {"method": "tap", "screen_size": [200, 100], "position": [200, 90]},
        {"camera_package": "com.huawei.camera; input keyevent HOME"},
        {"wake_position": [10, 20]},
        {"method": "keyevent", "screen_size": [200, 100], "position": [100, 90]},
        {"method": "tap", "screen_size": [200, 100], "position": [100, 90], "wake_delay_sec": 0},
    ],
)
def test_unsafe_shutter_configuration_fails_before_device_access(tmp_path, shutter):
    with pytest.raises(CaptureError):
        PhoneCapture(tmp_path, camera_shutter=shutter)


def test_pending_media_file_is_ignored_until_camera_renames_final_photo(environment, monkeypatch):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch)
    pending = "/sdcard/DCIM/Camera/.pending-123-IMG_20261025_013154.jpg"
    final = "/sdcard/DCIM/Camera/IMG_20261025_013154.jpg"
    raw = image_bytes(320, 180, image_format="JPEG")
    fake.on_shutter = lambda owner: owner.photos.update({pending: raw})

    def rename_when_ready(owner):
        if owner.list_count == 4:
            owner.photos[final] = owner.photos.pop(pending)

    fake.on_list = rename_when_ready
    result = PhoneCapture(root, camera_shutter=_tap_config()).capture_photo(reason="UI occlusion")
    assert result.meta["remote_photo"] == final
    assert result.raw_bytes == raw
    assert fake.shutters == 1
    assert not any(pending in " ".join(command) for command, _ in fake.calls)


@pytest.mark.parametrize("preview_failure", ["late_success", "subprocess_timeout"])
def test_woken_preview_timeout_falls_back_to_one_photo_without_second_wake(
    environment, monkeypatch, preview_failure
):
    root, _, clock = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    screenshot_count = 0

    def run(command, **kwargs):
        nonlocal screenshot_count
        if "screencap" in command:
            screenshot_count += 1
            if screenshot_count == 2 and preview_failure == "subprocess_timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        completed = fake(command, **kwargs)
        if "screencap" in command and screenshot_count == 2:
            clock.value += 3.1
        return completed

    monkeypatch.setattr("app.phone_capture.subprocess.run", run)
    result = PhoneCapture(root, camera_shutter=_tap_config()).capture()
    assert result.source == "camera_photo"
    assert result.meta["wake_count"] == 1
    assert result.meta["shutter_count"] == fake.shutters == 1
    assert fake.tap_positions == [[100, 50], [100, 90]]
    assert screenshot_count == 3


def test_uncertain_wake_command_timeout_cannot_cause_a_second_wake(environment, monkeypatch):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    wake_commands = 0

    def run(command, **kwargs):
        nonlocal wake_commands
        if command[-5:] == ["shell", "input", "tap", "100", "50"]:
            wake_commands += 1
            # 假设触控确实送达，但进程回报超时；相机仍未给出可用预览。
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return fake(command, **kwargs)

    monkeypatch.setattr("app.phone_capture.subprocess.run", run)
    with pytest.raises(CaptureError, match="仍黑屏"):
        PhoneCapture(root, camera_shutter=_tap_config()).capture()
    assert wake_commands == 1
    assert fake.shutters == 0


@pytest.mark.parametrize(
    "policy",
    [
        "deviceHasKeyguard=true\nshowing=true\nsecure=true",
        "showingAndNotOccluded=true\nsecure=true",
        "KeyguardStateMonitor\n  mIsShowing=true",
        "showing=false\nshowingAndNotOccluded=false\nmIsShowing=true",
    ],
)
@pytest.mark.parametrize("entry", ["capture", "capture_photo", "probe"])
def test_current_keyguard_blocks_capture_and_all_phone_input(
    environment, monkeypatch, policy, entry
):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    fake.keyguard_policy = policy
    phone = PhoneCapture(root, camera_shutter=_tap_config())
    if entry == "probe":
        report = phone.probe()
        assert report["available"] is False
        assert report["locked"] is True
        assert report["error"] == "手机已锁屏，请先解锁并打开原生相机"
    else:
        with pytest.raises(CaptureError, match="手机已锁屏，请先解锁并打开原生相机") as error:
            if entry == "capture":
                phone.capture()
            else:
                phone.capture_photo(reason="UI occlusion")
        assert error.value.details["phone_locked"] is True
    assert fake.shutters == 0
    assert fake.tap_positions == []
    assert not any("screencap" in command or "input" in command for command, _ in fake.calls)


@pytest.mark.parametrize(
    "policy",
    [
        "deviceHasKeyguard=true\nshowing=false\nshowingAndNotOccluded=false\nmIsShowing=false",
        "showing=false\nshowingAndNotOccluded=true\nsecure=true\nscreenState=SCREEN_STATE_ON\ninteractive=AWAKE\nKeyguardStateMonitor\n  mIsShowing=false\n  mInputRestricted=false",
        "showing=true\nshowingAndNotOccluded=true\nmIsShowing=false",
        "showing=false\nshowingAndNotOccluded=true",
    ],
)
def test_keyguard_capability_is_not_mistaken_for_currently_locked(environment, monkeypatch, policy):
    root, _, _ = environment
    fake = FakeAdb()
    fake.keyguard_policy = policy
    monkeypatch.setattr("app.phone_capture.subprocess.run", fake)
    assert PhoneCapture(root).capture(preview_only=True).source == "preview_screenshot"
    assert fake.shutters == 0


@pytest.mark.parametrize(
    "policy", ["deviceHasKeyguard=true\nsecure=true", "", "mIsShowing=true\nmIsShowing=false"]
)
def test_unknown_current_keyguard_state_stops_without_claiming_phone_is_locked(
    environment, monkeypatch, policy
):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch, asleep=True)
    fake.keyguard_policy = policy
    with pytest.raises(CaptureError, match="无法确认手机当前锁屏状态"):
        PhoneCapture(root, camera_shutter=_tap_config()).capture()
    assert fake.shutters == 0
    assert fake.tap_positions == []


def test_keyguard_is_rechecked_before_native_shutter_if_phone_locks_during_preparation(
    environment, monkeypatch
):
    root, _, _ = environment
    fake = _tap_fake(monkeypatch)
    fake.on_list = lambda owner: setattr(owner, "keyguard_policy", "mIsShowing=true")
    with pytest.raises(CaptureError, match="手机已锁屏"):
        PhoneCapture(root, camera_shutter=_tap_config()).capture_photo(reason="UI occlusion")
    assert fake.tap_positions == []
    assert fake.shutters == 0
