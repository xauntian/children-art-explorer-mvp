"""GUI 的原生相机只读适配器；始终截图，永不向手机发送触控或相机设置。

工作人员先解锁同一手机、打开华为相机并手动设置取景。这里复用已有截图
ROI 和 PNG 校验，但不使用 Camera2、原生快门、相机唤醒或拍照回退。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .errors import CaptureError
from .phone_capture import PhoneCapture, PhoneCaptureResult


class _ReadOnlyPhoneCapture(PhoneCapture):
    """白名单在实际执行前阻止任何写相机命令，包括意外的父类回退。"""

    _ALLOWED_COMMANDS = {
        ("devices", "-l"),
        ("shell", "dumpsys", "window", "policy"),
        ("shell", "dumpsys", "activity", "activities"),
        ("exec-out", "screencap", "-p"),
    }

    def _run(self, args: list[str], *, serial: str | None = None, timeout: float) -> bytes:
        if tuple(args) not in self._ALLOWED_COMMANDS:
            raise CaptureError("GUI 只允许读取手机状态与截屏", stage="capture")
        return super()._run(args, serial=serial, timeout=timeout)

    def capture(self, *, preview_only: bool = True) -> PhoneCaptureResult:
        # 同一轮只查询一次设备，并立即固定序列号，禁止后续切换手机。
        serial = self._select_serial()
        self.serial = serial
        # _foreground_camera 同时核对锁屏状态，无须父类再重复查询。
        self._foreground_camera(serial, time.monotonic() + self.command_timeout_sec * 2)
        return super()._capture_preview(serial, allow_wake=False)


class GuiCamera:
    """供 GUI / PhoneController 使用的同步接口，内部串行化 ADB 读取。

    prepare() / probe() 返回设备可用性报告，不抛出常规设备错误；capture()
    返回 PhoneCaptureResult，设备错误照常抛 CaptureError。
    close() 仅关闭本地适配器，不结束手机程序或共享 ADB 服务。
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        phone_config: dict[str, Any] | None = None,
    ) -> None:
        options = dict(phone_config or {})
        # 不继承旧配置里的 tap / wake_position，也不允许转入原生快门。
        options["fallback_photo"] = False
        options["camera_shutter"] = {"method": "keyevent", "camera_package": "com.huawei.camera"}
        self.phone = _ReadOnlyPhoneCapture(project_root, **options)
        self._lock = threading.RLock()
        self._closed = False
        self._last_report: dict[str, Any] = self._empty_report()

    @staticmethod
    def _empty_report() -> dict[str, Any]:
        return {
            "available": False,
            "serial": None,
            "camera_package": "com.huawei.camera",
            "screen_size": None,
            "error": None,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise CaptureError("GUI 手机采集器已关闭", stage="capture")

    @staticmethod
    def _report_for(capture: PhoneCaptureResult) -> dict[str, Any]:
        return {
            "available": True,
            "serial": capture.meta["serial"],
            "camera_package": "com.huawei.camera",
            "screen_size": [capture.meta["raw_width"], capture.meta["raw_height"]],
            "error": None,
        }

    def capture(self, *, preview_only: bool = True) -> PhoneCaptureResult:
        """获取当前相机截图；preview_only 参数兼容旧调用，绝不触发快门。"""
        with self._lock:
            self._ensure_open()
            capture = self.phone.capture(preview_only=True)
            result = PhoneCaptureResult(
                capture.raw_bytes,
                capture.source,
                capture.preview_image,
                {**capture.meta, "camera_package": "com.huawei.camera"},
            )
            self._last_report = self._report_for(result)
            return result

    def prepare(self) -> dict[str, Any]:
        """只读检查已授权手机、解锁状态、相机前台和取景截图。"""
        with self._lock:
            try:
                self.capture()
            except CaptureError as exc:
                self._last_report = {**self._empty_report(), "serial": self.phone.serial, "error": str(exc)}
            return dict(self._last_report)

    def probe(self) -> dict[str, Any]:
        return self.prepare()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._last_report = self._empty_report()
