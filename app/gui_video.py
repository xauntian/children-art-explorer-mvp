"""Memory-only video preview of the phone's existing display.

The native camera remains under the user's control. This stream never opens a
camera, sends input, reads UI XML, or captures screenshots. Frame age measures
local H.264 decode time using the monotonic clock.
"""

from __future__ import annotations

import os
import re
import secrets
import shlex
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CaptureError
from .phone_capture import PhoneCapture


@dataclass(frozen=True, slots=True)
class PreviewFrame:
    image: np.ndarray
    sequence: int
    age_ms: float


class ScreenVideoPreview:
    """Own one scrcpy display stream, independent of the photo capture lock."""

    def __init__(self, project_root, phone_config, options=None):
        self.project_root = Path(project_root).resolve()
        phone = dict(phone_config or {})
        self._phone = PhoneCapture(
            self.project_root,
            adb_path=phone.get("adb_path"),
            serial=phone.get("serial"),
            command_timeout_sec=phone.get("command_timeout_sec", 5),
        )
        settings = dict(options or {})
        allowed = {"server_path", "version", "max_size", "max_fps", "bit_rate", "startup_timeout_sec", "max_frame_age_ms"}
        if set(settings) - allowed:
            raise CaptureError("未知屏幕视频配置: " + ", ".join(sorted(set(settings) - allowed)), stage="capture")
        server_path = Path(settings.get("server_path", "tools/android/scrcpy-server-v5.0.1"))
        self.server_path = (self.project_root / server_path).resolve()
        self.version = settings.get("version", "5.0.1")
        if not isinstance(self.version, str) or not re.fullmatch(r"[\w.-]+", self.version):
            raise CaptureError("屏幕视频 version 无效", stage="capture")
        self.max_size = self._integer(settings.get("max_size", 1280), "max_size")
        self.max_fps = self._integer(settings.get("max_fps", 15), "max_fps")
        self.bit_rate = self._integer(settings.get("bit_rate", 4_000_000), "bit_rate")
        self.startup_timeout_sec = PhoneCapture._positive(settings.get("startup_timeout_sec", 12), "startup_timeout_sec")
        self.max_frame_age_ms = PhoneCapture._positive(settings.get("max_frame_age_ms", 1000), "max_frame_age_ms")
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._serial = None
        self._port = None
        self._remote_server = None
        self._remote_pid = None
        self._scid = None
        self._process = None
        self._socket = None
        self._thread = None
        self._log_thread = None
        self._initial_chunk = b""
        self._frame = None
        self._sequence = 0
        self._decoded_at = 0.0
        self._frame_times = deque(maxlen=60)
        self._logs = deque(maxlen=12)
        self._error = None
        self._ready = False
        self.cleanup_errors: list[str] = []

    @staticmethod
    def _integer(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CaptureError(f"屏幕视频 {name} 必须为正整数", stage="capture")
        return value

    def _adb(self, *args, timeout=None):
        return self._phone._run(list(args), serial=self._serial, timeout=timeout or self._phone.command_timeout_sec)

    @property
    def error(self) -> str | None:
        with self._condition:
            if self._error:
                return self._error
            if self._ready and self._decoded_at and (time.monotonic() - self._decoded_at) * 1000 > self.max_frame_age_ms:
                return "手机实时视频暂无新画面，等待手机画面更新"
            return None

    def _launch_command(self):
        arguments = [
            "app_process", "/", "com.genymobile.scrcpy.Server", self.version,
            f"scid={self._scid}", "tunnel_forward=true", "video_source=display",
            "video_codec=h264", f"max_size={self.max_size}", f"max_fps={self.max_fps}",
            f"video_bit_rate={self.bit_rate}", "audio=false", "control=false",
            "cleanup=false", "raw_stream=true",
        ]
        # The PID marker belongs to this unique server shell. exec keeps its PID
        # so shutdown can verify its cmdline before killing only this process.
        command = f"echo AMCC5000_PREVIEW_PID=$$; CLASSPATH={shlex.quote(self._remote_server)} exec {shlex.join(arguments)}"
        return [str(self._phone.adb_path), "-s", self._serial, "shell", command]

    def start(self, serial=None):
        """Start the display server and wait for its first decoded frame."""
        with self._lifecycle_lock:
            if self._ready:
                if self.error:
                    raise CaptureError(self.error, stage="capture")
                return self
            deadline = time.monotonic() + self.startup_timeout_sec
            try:
                if not self.server_path.is_file():
                    raise CaptureError(f"屏幕视频 server 不存在: {self.server_path}", stage="capture")
                import av  # noqa: F401

                if serial is not None:
                    if not isinstance(serial, str) or not serial.strip():
                        raise CaptureError("屏幕视频 serial 无效", stage="capture")
                    self._phone.serial = serial.strip()
                self._serial = self._phone._select_serial()
                self._stop.clear()
                with self._condition:
                    self._error = None
                    self._frame = None
                    self._sequence = 0
                    self._decoded_at = 0.0
                    self._frame_times.clear()
                    self._logs.clear()
                self.cleanup_errors = []
                self._scid = f"{secrets.randbits(31):08x}"
                self._remote_pid = None
                self._remote_server = f"/data/local/tmp/amcc5000-preview-{self._scid}.jar"
                self._adb("push", str(self.server_path), self._remote_server)
                self._port = int(self._adb("forward", "tcp:0", "localabstract:scrcpy_" + self._scid).strip())
                self._process = subprocess.Popen(
                    self._launch_command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self._log_thread = threading.Thread(target=self._read_logs, name=f"preview-log-{self._scid}", daemon=True)
                self._log_thread.start()
                self._connect(deadline)
                self._thread = threading.Thread(target=self._decode_loop, name=f"screen-video-{self._scid}", daemon=True)
                self._thread.start()
                with self._condition:
                    while self._frame is None:
                        if self._error:
                            raise CaptureError(self._error, stage="capture")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise CaptureError("手机实时视频等待首帧超时", stage="capture")
                        self._condition.wait(min(remaining, 0.2))
                    if self._error:
                        raise CaptureError(self._error, stage="capture")
                    self._ready = True
                return self
            except Exception as exc:
                message = str(exc)
                with self._condition:
                    self._error = f"手机实时视频启动失败: {message}"
                self._close_owned()
                if isinstance(exc, CaptureError):
                    raise
                raise CaptureError(self._error, stage="capture") from exc

    def _read_logs(self):
        try:
            for line in self._process.stdout:
                text = line.decode("utf-8", errors="replace").strip()
                marker = re.fullmatch(r"AMCC5000_PREVIEW_PID=(\d+)", text)
                with self._condition:
                    if marker:
                        self._remote_pid = int(marker.group(1))
                    elif text:
                        self._logs.append(text)
                    self._condition.notify_all()
        except (OSError, ValueError, AttributeError):
            pass

    def _connect(self, deadline):
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                with self._condition:
                    details = " ".join(self._logs)
                raise CaptureError(f"手机实时视频 server 已退出: {details or '请检查手机是否支持屏幕编码'}", stage="capture")
            try:
                if self._socket is None:
                    self._socket = socket.create_connection(("127.0.0.1", self._port), timeout=min(0.2, max(deadline - time.monotonic(), 0.001)))
                    self._socket.settimeout(0.2)
                self._initial_chunk = self._socket.recv(65536)
                if self._initial_chunk:
                    return
                self._socket.close()
                self._socket = None
            except TimeoutError:
                continue
            except OSError:
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
            self._stop.wait(min(0.05, max(0, deadline - time.monotonic())))
        raise CaptureError("手机实时视频连接超时", stage="capture")

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
                        # scrcpy emits no new frames for an unchanged display.
                        # Keep the stream alive so a later camera frame resumes
                        # immediately; readers still reject locally stale data.
                        continue
                    if not chunk:
                        raise OSError("USB 屏幕视频连接已关闭")
                for packet in decoder.parse(chunk):
                    for decoded in decoder.decode(packet):
                        image = decoded.to_ndarray(format="bgr24")
                        now = time.monotonic()
                        with self._condition:
                            self._frame = image
                            self._decoded_at = now
                            self._sequence += 1
                            self._frame_times.append(now)
                            self._condition.notify_all()
                chunk = b""
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._error = f"手机实时视频已中断: {exc}"
                    self._condition.notify_all()

    def latest_frame(self) -> PreviewFrame | None:
        """Return only the most recent frame; never return a stale/failed stream."""
        with self._condition:
            if not self._ready or self._stop.is_set() or self._error or self._frame is None:
                return None
            age_ms = (time.monotonic() - self._decoded_at) * 1000
            if not 0 <= age_ms <= self.max_frame_age_ms:
                return None
            return PreviewFrame(self._frame.copy(), self._sequence, age_ms)

    def report(self) -> dict[str, Any]:
        with self._condition:
            frame = self._frame
            height, width = frame.shape[:2] if frame is not None else (0, 0)
            duration = self._frame_times[-1] - self._frame_times[0] if len(self._frame_times) > 1 else 0
            fps = (len(self._frame_times) - 1) / duration if duration > 0 else 0.0
            fresh = frame is not None and 0 <= (time.monotonic() - self._decoded_at) * 1000 <= self.max_frame_age_ms
            return {"available": bool(self._ready and not self._stop.is_set() and not self._error and fresh),
                    "width": width, "height": height, "fps": round(fps, 1), "source": "screen_video", "error": self.error}

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
        if self._remote_pid is not None:
            try:
                cmdline = self._adb("shell", "cat", f"/proc/{self._remote_pid}/cmdline")
                if b"com.genymobile.scrcpy.Server" in cmdline and f"scid={self._scid}".encode() in cmdline:
                    self._adb("shell", "kill", str(self._remote_pid))
            except CaptureError as exc:
                # Socket closure normally exits scrcpy before this verification.
                self.cleanup_errors.append(str(exc))
            self._remote_pid = None
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.cleanup_errors.append(str(exc))
            if self._log_thread is not None:
                self._log_thread.join(timeout=1)
                self._log_thread = None
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._process = None
        for args in (
            ("forward", "--remove", f"tcp:{self._port}") if self._port is not None else None,
            ("shell", "rm", "-f", self._remote_server) if self._remote_server is not None else None,
        ):
            if args is not None:
                try:
                    self._adb(*args)
                except CaptureError as exc:
                    self.cleanup_errors.append(str(exc))
        self._port = self._remote_server = None
        self._frame = None

    def close(self):
        with self._lifecycle_lock:
            self._close_owned()
