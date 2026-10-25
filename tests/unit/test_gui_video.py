"""Display video ownership, freshness, and decoder tests without a phone."""

from __future__ import annotations

import io
import os
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.errors import CaptureError
from app.gui_video import PreviewFrame, ScreenVideoPreview


@pytest.fixture
def preview(tmp_path):
    server = tmp_path / "server"
    server.write_bytes(b"test server")
    return ScreenVideoPreview(tmp_path, {"serial": "PHONE"}, {"server_path": str(server)})


def publish(preview, *, value=42, age_ms=0):
    with preview._condition:
        preview._frame = np.full((48, 64, 3), value, np.uint8)
        preview._sequence += 1
        preview._decoded_at = time.monotonic() - age_ms / 1000
        preview._frame_times.append(preview._decoded_at)
        preview._condition.notify_all()


class FakeSocket:
    def __init__(self):
        self.closed = False
        self.shutdown_called = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, count):
        return b"codec data"

    def shutdown(self, flag):
        assert flag == socket.SHUT_RDWR
        self.shutdown_called = True

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.terminated = False
        self.stdout = io.BytesIO(b"AMCC5000_PREVIEW_PID=2468\n[server] INFO: Display 0\n")

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def wait(self, timeout):
        return 0


@pytest.fixture
def lifecycle(preview, monkeypatch):
    calls = []
    processes = []
    stream_socket = FakeSocket()

    def adb(args, **kwargs):
        calls.append(args)
        if args == ["devices", "-l"]:
            return b"List of devices attached\nPHONE\tdevice\n"
        if args[:2] == ["forward", "tcp:0"]:
            return b"31234\n"
        if args[:2] == ["shell", "cat"]:
            return f"app_process\0com.genymobile.scrcpy.Server\0scid={preview._scid}\0".encode()
        return b""

    def create_process(args, **kwargs):
        result = FakeProcess(args, **kwargs)
        processes.append(result)
        return result

    def warm_decoder():
        publish(preview)
        preview._stop.wait(2)

    monkeypatch.setattr(preview._phone, "_run", adb)
    monkeypatch.setattr("app.gui_video.subprocess.Popen", create_process)
    monkeypatch.setattr("app.gui_video.socket.create_connection", lambda *a, **kw: stream_socket)
    monkeypatch.setattr(preview, "_decode_loop", warm_decoder)
    return calls, processes, stream_socket


def test_start_uses_display_video_only_and_close_only_cleans_its_resources(preview, lifecycle):
    calls, processes, stream_socket = lifecycle
    assert preview.start(serial="PHONE") is preview
    assert preview.start() is preview
    assert len(processes) == 1
    command = processes[0].args[-1]
    for argument in ("video_source=display", "video_codec=h264", "raw_stream=true", "audio=false", "control=false", "max_size=1280", "max_fps=15", "video_bit_rate=4000000"):
        assert argument in command
    for forbidden in ("video_source=camera", "camera_zoom", "camera_id", "screencap", "uiautomator", "settings", "stayon", "force-stop", "keyevent"):
        assert forbidden not in command
        assert not any(forbidden in " ".join(call) for call in calls)
    assert processes[0].kwargs["stdout"] == subprocess.PIPE
    assert processes[0].kwargs["stdin"] == subprocess.DEVNULL
    assert processes[0].kwargs["creationflags"] == (subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert not (preview.project_root / "runs").exists()
    remote = next(call[2] for call in calls if call[0] == "push")
    assert remote.startswith("/data/local/tmp/amcc5000-preview-")
    frame = preview.latest_frame()
    assert isinstance(frame, PreviewFrame)
    assert frame.sequence == 1 and np.all(frame.image == 42)
    assert frame.age_ms >= 0
    assert preview.report()["source"] == "screen_video"
    assert preview.report()["available"]
    assert preview.report()["width"] == 64
    preview.close()
    assert stream_socket.closed and stream_socket.shutdown_called
    assert processes[0].terminated
    assert ["forward", "--remove", "tcp:31234"] in calls
    assert ["shell", "rm", "-f", remote] in calls
    assert ["shell", "kill", "2468"] in calls
    assert preview.latest_frame() is None
    assert not preview.report()["available"]
    before = len(calls)
    preview.close()
    assert len(calls) == before


def test_latest_returns_only_newest_image_and_an_independent_copy(preview):
    assert preview.latest_frame() is None
    preview._ready = True
    publish(preview, value=1)
    publish(preview, value=9)
    latest = preview.latest_frame()
    assert latest.sequence == 2 and np.all(latest.image == 9)
    latest.image[:] = 88
    assert np.all(preview.latest_frame().image == 9)


def test_stale_or_failed_stream_never_returns_cached_frame(preview):
    preview._ready = True
    publish(preview, age_ms=1100)
    assert preview.latest_frame() is None
    assert not preview.report()["available"]
    assert "暂无新画面" in preview.error
    publish(preview)
    preview._error = "USB stream lost"
    assert preview.latest_frame() is None
    assert preview.report()["error"] == "USB stream lost"


def test_report_computes_observed_fps_from_decode_timestamps(preview):
    preview._ready = True
    publish(preview)
    now = time.monotonic()
    preview._frame_times.extend([now - 0.2, now - 0.1, now])
    preview._frame_times.popleft()
    assert preview.report()["fps"] == 10


def test_failed_first_decode_cleans_owned_resources(preview, lifecycle, monkeypatch):
    calls, processes, stream_socket = lifecycle

    def fail_decoder():
        with preview._condition:
            preview._error = "cannot decode H.264"
            preview._condition.notify_all()

    monkeypatch.setattr(preview, "_decode_loop", fail_decoder)
    with pytest.raises(CaptureError, match="cannot decode"):
        preview.start()
    assert processes[0].terminated and stream_socket.closed
    assert ["forward", "--remove", "tcp:31234"] in calls
    assert preview.error and "启动失败" in preview.error
    assert not preview.report()["available"]


def test_cleanup_does_not_kill_reused_pid_or_unrelated_process(preview, monkeypatch):
    calls = []
    preview._serial = "PHONE"
    preview._remote_pid = 2468
    preview._scid = "01234567"

    def adb(*args, **kwargs):
        calls.append(args)
        return b"unrelated app_process\0scid=76543210\0"

    monkeypatch.setattr(preview, "_adb", adb)
    preview.close()
    assert calls == [("shell", "cat", "/proc/2468/cmdline")]


def test_decoder_eof_records_disconnect(preview, monkeypatch):
    preview._socket = SimpleNamespace(recv=lambda count: b"")
    decoder = SimpleNamespace(thread_count=0)
    monkeypatch.setattr("av.CodecContext", SimpleNamespace(create=lambda *a: decoder))
    preview._decode_loop()
    assert "已中断" in preview.error
    assert "连接已关闭" in preview.error


def test_static_display_timeouts_keep_stream_alive_and_next_frame_recovers(preview, monkeypatch):
    receives = []
    preview._ready = True
    publish(preview, age_ms=1100)
    assert preview.latest_frame() is None
    assert "暂无新画面" in preview.error

    def receive(count):
        receives.append(count)
        if len(receives) <= 3:
            assert preview._error is None
            raise TimeoutError
        return b"new h264 data"

    def decode(packet):
        preview._stop.set()
        return [SimpleNamespace(to_ndarray=lambda format: np.full((48, 64, 3), 9, np.uint8))]

    preview._socket = SimpleNamespace(recv=receive)
    decoder = SimpleNamespace(thread_count=0, parse=lambda data: [data], decode=decode)
    monkeypatch.setattr("av.CodecContext", SimpleNamespace(create=lambda *a: decoder))
    preview._decode_loop()
    preview._stop.clear()
    assert len(receives) == 4
    assert preview._error is None
    assert preview.error is None
    frame = preview.latest_frame()
    assert frame.sequence == 2 and np.all(frame.image == 9)
    assert preview.report()["available"] is True


def test_decoder_publishes_bgr_frames_without_queue(preview, monkeypatch):
    calls = []

    def image(format):
        calls.append(format)
        return np.ones((48, 64, 3), np.uint8)

    decoded = SimpleNamespace(to_ndarray=image)

    def decode(packet):
        preview._stop.set()
        return [decoded, decoded]

    decoder = SimpleNamespace(thread_count=0, parse=lambda data: [data], decode=decode)
    preview._initial_chunk = b"h264"
    monkeypatch.setattr("av.CodecContext", SimpleNamespace(create=lambda *a: decoder))
    preview._decode_loop()
    assert decoder.thread_count == 1
    assert calls == ["bgr24", "bgr24"]
    assert preview._sequence == 2
    assert preview._frame.shape == (48, 64, 3)
    assert abs(time.monotonic() - preview._decoded_at) < 0.1


@pytest.mark.parametrize("options", [{"unknown": 1}, {"max_fps": 0}, {"max_size": True}, {"bit_rate": -2}, {"max_frame_age_ms": 0}, {"version": "5; exit"}])
def test_invalid_options_fail_before_device_access(tmp_path, options):
    with pytest.raises(CaptureError):
        ScreenVideoPreview(tmp_path, {}, options)


def test_missing_server_does_not_touch_device(tmp_path, monkeypatch):
    preview = ScreenVideoPreview(tmp_path, {})
    monkeypatch.setattr(preview, "_adb", lambda *a, **kw: pytest.fail("must not call ADB"))
    with pytest.raises(CaptureError, match="server 不存在"):
        preview.start()
