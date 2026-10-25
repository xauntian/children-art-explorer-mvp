"""Fresh frame, disconnection, and owned-resource tests; no device interaction."""

from __future__ import annotations

import socket
import threading
import time

import cv2
import numpy as np
import pytest

from app.errors import CaptureError
from app.phone_stream import PhoneStreamCapture


@pytest.fixture
def camera(tmp_path):
    server = tmp_path / "server"
    server.write_bytes(b"test server")
    return PhoneStreamCapture(tmp_path, serial="PHONE", stream={"server_path": str(server), "fresh_frame_timeout_sec": 0.06}, camera_shutter={"camera_package": "com.huawei.camera"})


def ready(camera):
    camera._ready = True
    camera._serial = "PHONE"
    camera.startup_sec = 2.1
    publish(camera, 1)
    return camera


def publish(camera, value=72, *, decoded_ns=None):
    with camera._condition:
        camera._frame = np.full((48, 64, 3), value, np.uint8)
        camera._sequence += 1
        camera._decode_ns = decoded_ns or time.perf_counter_ns()
        camera._condition.notify_all()


def later(action, delay=0.01):
    timer = threading.Timer(delay, action)
    timer.start()
    return timer


def test_capture_needs_explicit_start(camera):
    with pytest.raises(CaptureError, match="start"):
        camera.capture()


def test_capture_waits_for_post_trigger_frame_without_adb_or_photo_lookup(camera, monkeypatch):
    ready(camera)
    monkeypatch.setattr(camera, "_run", lambda *a, **kw: pytest.fail("capture must not call ADB"))
    timer = later(lambda: publish(camera, 120))
    try:
        result = camera.capture()
    finally:
        timer.join()
    assert result.source == "camera_stream"
    assert result.meta["frame_sequence"] > result.meta["trigger_sequence"]
    assert result.meta["frame_local_decode_monotonic_ns"] >= result.meta["trigger_monotonic_ns"]
    assert result.meta["timing_clock"] == "perf_counter_ns"
    assert result.meta["shutter_count"] == 0
    assert result.meta["preview_roi"] is None
    assert result.meta["startup_included_in_capture"] is False
    assert result.meta["stream_startup_sec"] == 2.1
    assert result.meta["frame_age_scope"] == "local_decode_only"
    assert result.meta["camera_zoom"] == 2
    decoded = cv2.imdecode(np.frombuffer(result.raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 64, 3)
    assert np.all(decoded == 120)


def test_existing_cached_frame_is_never_returned(camera):
    ready(camera)
    with pytest.raises(CaptureError, match="未复用旧帧"):
        camera.capture()


def test_increasing_sequence_with_old_decode_timestamp_is_rejected(camera):
    ready(camera)
    stale_ns = time.perf_counter_ns() - 1_000_000_000
    timer = later(lambda: publish(camera, decoded_ns=stale_ns))
    try:
        with pytest.raises(CaptureError, match="未复用旧帧"):
            camera.capture()
    finally:
        timer.join()


def test_stream_failure_wakes_capture_and_cannot_trigger_native_photo(camera):
    ready(camera)

    def fail_stream():
        with camera._condition:
            camera._decode_error = "stream disconnected"
            camera._condition.notify_all()

    timer = later(fail_stream)
    try:
        with pytest.raises(CaptureError, match="disconnected"):
            camera.capture()
    finally:
        timer.join()
    with pytest.raises(CaptureError, match="不触发原生拍照"):
        camera.capture_photo(reason="anything")


def test_slow_encoding_does_not_deliver_expired_frame(camera, monkeypatch):
    ready(camera)
    camera.max_frame_age_ms = 5
    original = cv2.imencode

    def slow_encode(*args):
        time.sleep(0.015)
        return original(*args)

    monkeypatch.setattr("app.phone_stream.cv2.imencode", slow_encode)
    timer = later(lambda: publish(camera))
    try:
        with pytest.raises(CaptureError, match="编码后已过期"):
            camera.capture()
    finally:
        timer.join()


def test_close_wakes_waiting_capture(camera):
    ready(camera)
    timer = later(camera.close)
    try:
        with pytest.raises(CaptureError, match="已关闭"):
            camera.capture()
    finally:
        timer.join()


@pytest.mark.parametrize("options", [
    {"unknown": 1}, {"fps": True}, {"fps": 0}, {"bit_rate": -1},
    {"zoom": float("nan")}, {"max_frame_age_ms": 0}, {"size": "abc"},
    {"size": [1, 0]}, {"camera_id": "0;reboot"}, {"version": "5\nboom"},
])
def test_invalid_stream_options_rejected(tmp_path, options):
    with pytest.raises(CaptureError):
        PhoneStreamCapture(tmp_path, stream=options)


def test_probe_is_read_only_and_does_not_connect_camera(camera, monkeypatch):
    calls = []

    def adb(args, **kwargs):
        calls.append(args)
        return b"List of devices attached\nPHONE\tdevice\n"

    monkeypatch.setattr(camera, "_run", adb)
    report = camera.probe()
    assert report["available"] is True
    assert report["stream_ready"] is False
    assert calls == [["devices", "-l"]]


class FakeSocket:
    def __init__(self):
        self.closed = False
        self.shutdown_called = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, count):
        return b"initial codec data"

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

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        assert timeout > 0
        return 0


@pytest.fixture
def lifecycle(camera, monkeypatch):
    calls = []
    power = ["5"]
    procs = []
    sock = FakeSocket()

    def adb(args, **kwargs):
        calls.append(args)
        if args == ["devices", "-l"]:
            return b"List of devices attached\nPHONE\tdevice\n"
        if args == ["shell", "dumpsys", "window", "policy"]:
            return b"mIsShowing=false"
        if args[:5] == ["shell", "settings", "get", "global", "stay_on_while_plugged_in"]:
            return power[0].encode()
        if args == ["shell", "svc", "power", "stayon", "usb"]:
            power[0] = "2"
        if args[:5] == ["shell", "settings", "put", "global", "stay_on_while_plugged_in"]:
            power[0] = args[5]
        if args[:2] == ["forward", "tcp:0"]:
            return b"31234\n"
        return b""

    def process(args, **kwargs):
        result = FakeProcess(args, **kwargs)
        procs.append(result)
        return result

    def warm_decoder():
        # Simulate the decoder delivering a stable one-second history.
        with camera._condition:
            camera._first_decode_ns = time.perf_counter_ns() - 1_100_000_000
            camera._decode_ns = time.perf_counter_ns()
            camera._sequence = 34
            camera._frame = np.zeros((48, 64, 3), np.uint8)
            camera._condition.notify_all()
        camera._stop.wait(2)

    monkeypatch.setattr(camera, "_run", adb)
    monkeypatch.setattr("app.phone_stream.subprocess.Popen", process)
    monkeypatch.setattr("app.phone_stream.socket.create_connection", lambda *a, **kw: sock)
    monkeypatch.setattr(camera, "_decode_loop", warm_decoder)
    return calls, power, procs, sock


def test_start_pushes_owned_server_uses_camera_zoom_and_restores_exact_setting(camera, lifecycle):
    calls, power, procs, sock = lifecycle
    with camera as active:
        assert active is camera
        assert camera._ready
        assert power[0] == "2"
        assert camera._sequence >= 30
        command = procs[0].args
        assert "camera_zoom=2" in command
        assert "camera_id=0" in command
        assert "camera_size=2048x1536" in command
        assert "raw_stream=true" in command
        assert "video_source=camera" in command
        assert "audio=false" in command
        assert "control=false" in command
        assert ["shell", "am", "force-stop", "com.huawei.camera"] in calls
        pushes = [call for call in calls if call[0] == "push"]
        assert len(pushes) == 1
        owned_remote = pushes[0][2]
        assert owned_remote.startswith("/data/local/tmp/amcc5000-camera-")
        assert camera.start() is camera  # no duplicate server
        assert len(procs) == 1
    assert power[0] == "5"
    assert sock.closed and sock.shutdown_called
    assert procs[0].terminated
    assert ["forward", "--remove", "tcp:31234"] in calls
    assert ["shell", "rm", "-f", owned_remote] in calls
    assert not camera.cleanup_errors
    assert not any("DCIM" in " ".join(call) or "killall" in call for call in calls)
    assert ["shell", "dumpsys", "window", "policy"] not in calls
    call_count = len(calls)
    camera.close()
    assert len(calls) == call_count


def test_failed_start_cleans_owned_resources_and_restores_phone(camera, lifecycle, monkeypatch):
    calls, power, procs, sock = lifecycle
    camera.startup_timeout_sec = 0.05

    def fail_decoder():
        with camera._condition:
            camera._decode_error = "cannot decode camera"
            camera._condition.notify_all()

    monkeypatch.setattr(camera, "_decode_loop", fail_decoder)
    with pytest.raises(CaptureError, match="cannot decode"):
        camera.start()
    assert power[0] == "5"
    assert procs[0].terminated
    assert sock.closed
    assert ["forward", "--remove", "tcp:31234"] in calls
    assert not camera._ready


def test_decoder_eof_marks_failure_and_wakes_consumers(camera, monkeypatch):
    class EofSocket:
        def recv(self, count):
            return b""

    camera._socket = EofSocket()
    camera._decode_loop()
    assert "stream closed" in camera._decode_error


def test_missing_server_does_not_mutate_phone(camera, monkeypatch):
    camera.server_path.unlink()
    monkeypatch.setattr(camera, "_run", lambda *a, **kw: pytest.fail("must validate server first"))
    with pytest.raises(CaptureError, match="server 不存在"):
        camera.start()
