"""A persistent, UI-free USB camera stream with fresh-frame capture.

Starting the stream is an explicit preparation step. Capturing never starts a
camera, takes a still photograph, searches DCIM, or falls back to another input.
Frame age measures local decode time, not the sensor exposure timestamp.
"""

from __future__ import annotations

import os
import re
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from .errors import CaptureError
from .phone_capture import PhoneCapture, PhoneCaptureResult
from .storage import atomic_write_json, utc_now_iso


class PhoneStreamCapture(PhoneCapture):
    """Own one scrcpy camera server, its ADB forward, and its decoder thread."""

    def __init__(self, project_root=None, *, stream: dict[str, Any] | None = None, **phone):
        super().__init__(project_root, **phone)
        options = dict(stream or {})
        known = {
            "server_path", "version", "camera_id", "size", "fps", "zoom", "bit_rate",
            "startup_timeout_sec", "fresh_frame_timeout_sec", "max_frame_age_ms",
        }
        if set(options) - known:
            raise CaptureError("未知串流配置: " + ", ".join(sorted(set(options) - known)), stage="capture")
        path = Path(options.get("server_path", "tools/android/scrcpy-server-v5.0.1"))
        self.server_path = (self.project_root / path).resolve() if not path.is_absolute() else path.resolve()
        self.version = self._option_text(options.get("version", "5.0.1"), "version")
        self.camera_id = self._option_text(options.get("camera_id", "0"), "camera_id")
        size = options.get("size", [2048, 1536])
        if isinstance(size, str):
            parts = size.split("x")
            try:
                size = [int(part) for part in parts] if len(parts) == 2 else None
            except ValueError:
                size = None
        self.size = self._validate_screen_size(size)
        if self.size is None:
            raise CaptureError("stream.size 必须为 [width, height] 或 WIDTHxHEIGHT", stage="capture")
        self.fps = self._integer(options.get("fps", 30), "fps")
        self.bit_rate = self._integer(options.get("bit_rate", 16_000_000), "bit_rate")
        self.zoom = self._positive(options.get("zoom", 2), "stream.zoom")
        self.startup_timeout_sec = self._positive(options.get("startup_timeout_sec", 15), "startup_timeout_sec")
        self.fresh_frame_timeout_sec = self._positive(options.get("fresh_frame_timeout_sec", 1), "fresh_frame_timeout_sec")
        self.max_frame_age_ms = self._positive(options.get("max_frame_age_ms", 150), "max_frame_age_ms")
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._socket = None
        self._initial_chunk = b""
        self._process = None
        self._log = None
        self._serial = None
        self._port = None
        self._remote_server = None
        self._power_original = None
        self._power_changed = False
        self._power_record = None
        self._frame = None
        self._sequence = 0
        self._decode_ns = 0
        self._first_decode_ns = 0
        self._decode_error = None
        self._ready = False
        self.startup_sec = None
        self.cleanup_errors: list[str] = []
        self.log_path: Path | None = None

    @staticmethod
    def _option_text(value, name):
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
            raise CaptureError(f"stream.{name} 必须为有效的单值参数", stage="capture")
        return value

    @staticmethod
    def _integer(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CaptureError(f"stream.{name} 必须为正整数", stage="capture")
        return value

    def _adb(self, *args, timeout=None):
        return self._run(list(args), serial=self._serial, timeout=timeout or self.command_timeout_sec)

    def start(self):
        """Start once and wait for >=30 frames and >=1 second of decoded video."""
        with self._lifecycle_lock:
            if self._ready:
                with self._condition:
                    if self._decode_error:
                        raise CaptureError(self._decode_error, stage="capture")
                return self
            started = time.monotonic()
            deadline = started + self.startup_timeout_sec
            try:
                if not self.server_path.is_file():
                    raise CaptureError(f"串流 server 不存在: {self.server_path}", stage="capture")
                # Import before changing anything on the phone.
                import av  # noqa: F401

                self._serial = self._select_serial()
                # Camera2 enforces camera access itself. This source never reads
                # the phone screen or sends unlock gestures, so native-camera UI
                # keyguard preconditions do not apply to the direct camera feed.
                self._stop.clear()
                self._decode_error = None
                self._sequence = self._decode_ns = self._first_decode_ns = 0
                self._frame = None
                self.cleanup_errors = []
                runtime = self.project_root / "runs" / "runtime"
                runtime.mkdir(parents=True, exist_ok=True)
                scid = f"{secrets.randbits(31):08x}"
                self._remote_server = f"/data/local/tmp/amcc5000-camera-{scid}.jar"
                self.log_path = runtime / f"phone-stream-{scid}.log"
                self._power_record = runtime / f"phone-stream-power-{scid}.json"
                original = self._adb("shell", "settings", "get", "global", "stay_on_while_plugged_in").decode().strip()
                if original != "null" and not original.isdigit():
                    raise CaptureError("无法读取手机原有常亮设置", stage="capture")
                self._power_original = original
                atomic_write_json(self._power_record, {"serial": self._serial, "original": original, "restored": False, "at": utc_now_iso()})
                # Mark ownership before invocation: a timed-out command may have applied.
                self._power_changed = True
                self._adb("shell", "svc", "power", "stayon", "usb")
                self._adb("push", str(self.server_path), self._remote_server)
                # The configured native camera is the only app that may be stopped.
                self._adb("shell", "am", "force-stop", self.camera_shutter["camera_package"])
                self._port = int(self._adb("forward", "tcp:0", "localabstract:scrcpy_" + scid).strip())
                args = [str(self.adb_path), "-s", self._serial, "shell", f"CLASSPATH={self._remote_server}", "app_process", "/", "com.genymobile.scrcpy.Server", self.version,
                        "scid=" + scid, "tunnel_forward=true", "video_source=camera", "camera_id=" + self.camera_id,
                        f"camera_size={self.size[0]}x{self.size[1]}", f"camera_fps={self.fps}", f"camera_zoom={self.zoom:g}",
                        "video_codec=h264", f"video_bit_rate={self.bit_rate}", "audio=false", "control=false", "cleanup=false", "raw_stream=true"]
                self._log = self.log_path.open("wb")
                self._process = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT,
                                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                while time.monotonic() < deadline:
                    if self._process.poll() is not None:
                        raise CaptureError(f"相机串流启动失败，请查看 {self.log_path}", stage="capture")
                    try:
                        if self._socket is None:
                            self._socket = socket.create_connection(("127.0.0.1", self._port), timeout=min(0.2, max(deadline - time.monotonic(), 0.001)))
                            self._socket.settimeout(0.2)
                        self._initial_chunk = self._socket.recv(65536)
                        if self._initial_chunk:
                            break
                        self._socket.close()
                        self._socket = None
                        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                    except TimeoutError:
                        # ADB forwarding may connect before Camera2 emits data.
                        # Keep a live connection while the camera initializes.
                        continue
                    except OSError:
                        if self._socket is not None:
                            self._socket.close()
                            self._socket = None
                        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                else:
                    raise CaptureError("相机串流连接超时", stage="capture")
                self._thread = threading.Thread(target=self._decode_loop, name=f"phone-stream-{scid}", daemon=True)
                self._thread.start()
                with self._condition:
                    while True:
                        if self._decode_error:
                            raise CaptureError(self._decode_error, stage="capture")
                        stable = self._sequence >= 30 and self._decode_ns - self._first_decode_ns >= 1_000_000_000
                        if stable:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise CaptureError("相机串流预热超时", stage="capture")
                        self._condition.wait(min(remaining, 0.2))
                self.startup_sec = round(time.monotonic() - started, 6)
                self._ready = True
                return self
            except Exception as exc:
                self._close_owned()
                if isinstance(exc, CaptureError):
                    raise
                raise CaptureError(f"相机串流启动失败: {exc}", stage="capture") from exc

    def _decode_loop(self):
        try:
            import av

            decoder = av.CodecContext.create("h264", "r")
            decoder.thread_count = 1
            chunk = self._initial_chunk
            self._initial_chunk = b""
            while not self._stop.is_set():
                if not chunk:
                    try:
                        chunk = self._socket.recv(65536)
                    except TimeoutError:
                        continue
                    if not chunk:
                        raise OSError("USB video stream closed")
                for packet in decoder.parse(chunk):
                    for decoded in decoder.decode(packet):
                        image = decoded.to_ndarray(format="bgr24")
                        decoded_ns = time.perf_counter_ns()
                        with self._condition:
                            self._frame = image
                            self._decode_ns = decoded_ns
                            self._sequence += 1
                            if not self._first_decode_ns:
                                self._first_decode_ns = decoded_ns
                            self._condition.notify_all()
                chunk = b""
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._decode_error = f"相机串流已中断: {exc}"
                    self._condition.notify_all()

    def capture(self, *, preview_only=False):
        """Wait for a frame decoded after this call; never reuse a cached frame."""
        trigger_ns = time.perf_counter_ns()
        deadline = time.monotonic() + self.fresh_frame_timeout_sec
        with self._condition:
            if not self._ready or self._stop.is_set():
                raise CaptureError("相机串流尚未启动，请在采集前调用 start()", stage="capture")
            trigger_sequence = self._sequence
            while True:
                if self._decode_error:
                    raise CaptureError(self._decode_error, stage="capture")
                if self._stop.is_set():
                    raise CaptureError("相机串流已关闭", stage="capture")
                now_ns = time.perf_counter_ns()
                age_ms = (now_ns - self._decode_ns) / 1_000_000
                if self._sequence > trigger_sequence and self._decode_ns >= trigger_ns and 0 <= age_ms <= self.max_frame_age_ms:
                    image = self._frame
                    decoded_ns = self._decode_ns
                    sequence = self._sequence
                    selected_ns = now_ns
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CaptureError("等待手机新鲜视频帧超时，未复用旧帧", stage="capture")
                self._condition.wait(remaining)
        image = image.copy()
        encode_started_ns = time.perf_counter_ns()
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise CaptureError("视频帧 PNG 编码失败", stage="capture")
        finished_ns = time.perf_counter_ns()
        returned_age_ms = (finished_ns - decoded_ns) / 1_000_000
        if returned_age_ms > self.max_frame_age_ms:
            raise CaptureError("视频帧编码后已过期，未输出旧帧", stage="capture")
        with self._condition:
            if self._decode_error or self._stop.is_set():
                raise CaptureError(self._decode_error or "相机串流已关闭", stage="capture")
        height, width = image.shape[:2]
        meta = {
            "raw_format": "png", "raw_suffix": ".png", "raw_width": width, "raw_height": height,
            "width": width, "height": height, "orientation": "landscape" if width > height else "portrait" if width < height else "square",
            "exif_orientation": 1, "serial": self._serial, "source": "camera_stream",
            "preview_roi": None, "preview_screen_size": None, "preview_width": width, "preview_height": height,
            "camera_id": self.camera_id, "camera_zoom": self.zoom, "camera_fps": self.fps,
            "video_codec": "h264", "video_bit_rate": self.bit_rate, "frame_sequence": sequence,
            "trigger_sequence": trigger_sequence, "trigger_monotonic_ns": trigger_ns, "frame_local_decode_monotonic_ns": decoded_ns,
            "timing_clock": "perf_counter_ns",
            "frame_age_scope": "local_decode_only", "frame_local_decode_age_ms": round((selected_ns - decoded_ns) / 1_000_000, 3),
            "frame_local_decode_age_at_return_ms": round(returned_age_ms, 3),
            "frame_wait_sec": round((selected_ns - trigger_ns) / 1_000_000_000, 6),
            "encode_sec": round((finished_ns - encode_started_ns) / 1_000_000_000, 6),
            "capture_elapsed_sec": round((finished_ns - trigger_ns) / 1_000_000_000, 6),
            "stream_startup_sec": self.startup_sec, "startup_included_in_capture": False,
            "shutter_count": 0, "wake_count": 0,
        }
        return PhoneCaptureResult(encoded.tobytes(), "camera_stream", image, meta)

    def capture_photo(self, *, reason):
        raise CaptureError("串流模式不触发原生拍照；请恢复视频连接后重试", stage="capture")

    def probe(self):
        """Read-only inventory; do not start a camera or consume a frame."""
        report = {"available": False, "source": "camera_stream", "server_path": str(self.server_path),
                  "server_exists": self.server_path.is_file(), "camera_id": self.camera_id,
                  "size": list(self.size), "fps": self.fps, "zoom": self.zoom, "stream_ready": self._ready}
        try:
            devices = self.list_devices()
            report["devices"] = devices
            report["serial"] = self._select_serial(devices)
            report["available"] = report["server_exists"]
            with self._condition:
                report["stream_error"] = self._decode_error
                if self._decode_error:
                    report["available"] = False
        except CaptureError as exc:
            report["error"] = exc.message
        return report

    def _close_owned(self):
        self._ready = False
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=2)
            except OSError as exc:
                self.cleanup_errors.append(str(exc))
            self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._port is not None:
            try:
                self._adb("forward", "--remove", f"tcp:{self._port}")
            except CaptureError as exc:
                self.cleanup_errors.append(exc.message)
            self._port = None
        if self._remote_server is not None:
            try:
                self._adb("shell", "rm", "-f", self._remote_server)
            except CaptureError as exc:
                self.cleanup_errors.append(exc.message)
            self._remote_server = None
        if self._power_changed:
            try:
                if self._power_original == "null":
                    self._adb("shell", "settings", "delete", "global", "stay_on_while_plugged_in")
                else:
                    self._adb("shell", "settings", "put", "global", "stay_on_while_plugged_in", self._power_original)
                restored = self._adb("shell", "settings", "get", "global", "stay_on_while_plugged_in").decode().strip()
                verified = restored == self._power_original
                atomic_write_json(self._power_record, {"serial": self._serial, "original": self._power_original, "restored": verified, "restored_value": restored, "at": utc_now_iso()})
                if not verified:
                    self.cleanup_errors.append("手机常亮设置未恢复到原值")
                else:
                    self._power_changed = False
            except (CaptureError, OSError) as exc:
                self.cleanup_errors.append(str(exc))

    def close(self):
        with self._lifecycle_lock:
            self._close_owned()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
