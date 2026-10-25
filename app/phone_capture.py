"""ADB 手机采集：优先读取相机预览，仅在预览不可用时拍摄一张新照片。

本模块不启动或切换相机、不修改取景设置，也不删除手机照片。调用前由
工作人员打开原生相机并固定取景。所有 ADB 调用使用项目内的显式路径。
"""

from __future__ import annotations

import io
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import CaptureError


@dataclass(frozen=True, slots=True)
class PhoneCaptureResult:
    """原始文件字节与供后续 OpenCV 使用的 BGR 图像。

    截屏成功时 raw_bytes 保留完整原始 PNG，preview_image 只应用配置的
    预览 ROI。原生照片回退时 raw_bytes 保留原始照片，preview_image 应用
    EXIF 方向但不应用屏幕 ROI；两者都未做作品识别或缩放。
    """

    raw_bytes: bytes
    source: str
    preview_image: np.ndarray
    meta: dict[str, Any]


class PhoneCapture:
    """通过唯一已授权设备采集；capture 每次最多触发一次原生快门。"""

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        adb_path: str | Path | None = None,
        serial: str | None = None,
        preview_roi: list[float] | tuple[float, float, float, float] | None = None,
        preview_screen_size: list[int] | tuple[int, int] | None = None,
        camera_shutter: dict[str, Any] | None = None,
        screenshot_timeout_sec: float = 3.0,
        fallback_photo: bool = True,
        command_timeout_sec: float = 5.0,
        photo_timeout_sec: float = 20.0,
        poll_interval_sec: float = 0.25,
        photo_dir: str = "/sdcard/DCIM/Camera",
    ) -> None:
        self.project_root = (
            Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
        )
        selected_path = Path(adb_path or "tools/android/platform-tools/adb.exe")
        self.adb_path = (
            selected_path if selected_path.is_absolute() else self.project_root / selected_path
        ).resolve()
        if serial is not None and (not isinstance(serial, str) or not serial.strip()):
            raise CaptureError("手机 serial 必须是非空字符串或 null", stage="capture")
        self.serial = serial.strip() if serial else None
        self.preview_roi = self._validate_roi(preview_roi)
        self.preview_screen_size = self._validate_screen_size(preview_screen_size)
        self.screenshot_timeout_sec = self._positive(
            screenshot_timeout_sec, "screenshot_timeout_sec"
        )
        self.command_timeout_sec = self._positive(command_timeout_sec, "command_timeout_sec")
        self.photo_timeout_sec = self._positive(photo_timeout_sec, "photo_timeout_sec")
        self.poll_interval_sec = self._positive(poll_interval_sec, "poll_interval_sec")
        self.camera_shutter = self._validate_shutter(camera_shutter)
        if not isinstance(fallback_photo, bool):
            raise CaptureError("fallback_photo 必须是布尔值", stage="capture")
        self.fallback_photo = fallback_photo
        if (
            not isinstance(photo_dir, str)
            or not photo_dir.startswith("/")
            or any(char in photo_dir for char in ("\x00", "\r", "\n"))
            or ".." in PurePosixPath(photo_dir).parts
        ):
            raise CaptureError("photo_dir 必须是手机中的绝对目录", stage="capture")
        self.photo_dir = photo_dir.rstrip("/") or "/"

    @staticmethod
    def _positive(value: float, name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CaptureError(f"{name} 必须是正数", stage="capture") from exc
        if isinstance(value, bool) or not math.isfinite(parsed) or parsed <= 0:
            raise CaptureError(f"{name} 必须是正数", stage="capture")
        return parsed

    @staticmethod
    def _validate_shutter(value: dict[str, Any] | None) -> dict[str, Any]:
        if value is not None and not isinstance(value, dict):
            raise CaptureError("camera_shutter 必须是对象或 null", stage="capture")
        config = dict(value or {})
        allowed = {
            "method",
            "camera_package",
            "screen_size",
            "position",
            "wake_position",
            "wake_delay_sec",
        }
        if set(config) - allowed:
            raise CaptureError("camera_shutter 包含未知字段", stage="capture")
        method = config.get("method", "keyevent")
        package = config.get("camera_package", "com.huawei.camera")
        if method not in ("keyevent", "tap"):
            raise CaptureError("camera_shutter.method 必须为 keyevent 或 tap", stage="capture")
        if not isinstance(package, str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", package
        ):
            raise CaptureError(
                "camera_shutter.camera_package 必须是有效的相机包名", stage="capture"
            )
        size = PhoneCapture._validate_screen_size(config.get("screen_size"))

        def point(name: str) -> list[int] | None:
            selected = config.get(name)
            if selected is None:
                return None
            if (
                size is None
                or not isinstance(selected, (list, tuple))
                or len(selected) != 2
                or any(isinstance(part, bool) or not isinstance(part, int) for part in selected)
                or not 0 <= selected[0] < size[0]
                or not 0 <= selected[1] < size[1]
            ):
                raise CaptureError(
                    f"camera_shutter.{name} 必须在已标定 screen_size 内", stage="capture"
                )
            return list(selected)

        position = point("position")
        wake_position = point("wake_position")
        if method == "tap" and (size is None or position is None):
            raise CaptureError("tap 快门必须显式配置 screen_size 与 position", stage="capture")
        if method == "keyevent" and position is not None:
            raise CaptureError("keyevent 快门不使用 position", stage="capture")
        return {
            "method": method,
            "camera_package": package,
            "screen_size": list(size) if size is not None else None,
            "position": position,
            "wake_position": wake_position,
            "wake_delay_sec": PhoneCapture._positive(
                config.get("wake_delay_sec", 0.7), "wake_delay_sec"
            ),
        }

    @staticmethod
    def _validate_screen_size(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part <= 0 for part in value
            )
        ):
            raise CaptureError(
                "preview_screen_size 必须为 [width, height] 正整数或 null", stage="capture"
            )
        return tuple(value)

    @staticmethod
    def _validate_roi(value: Any) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise CaptureError(
                "preview_roi 必须为 [left, top, right, bottom] 或 null", stage="capture"
            )
        try:
            coords = tuple(float(part) for part in value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CaptureError("preview_roi 坐标必须是 0 到 1 的数字", stage="capture") from exc
        if (
            any(isinstance(part, bool) for part in value)
            or not all(math.isfinite(part) and 0 <= part <= 1 for part in coords)
            or coords[0] >= coords[2]
            or coords[1] >= coords[3]
        ):
            raise CaptureError(
                "preview_roi 边界必须满足 0≤left<right≤1、0≤top<bottom≤1", stage="capture"
            )
        return coords

    def _run(self, args: list[str], *, serial: str | None = None, timeout: float) -> bytes:
        if not self.adb_path.is_file():
            raise CaptureError(f"项目 ADB 不存在: {self.adb_path}", stage="capture")
        command = [str(self.adb_path)]
        if serial is not None:
            command.extend(["-s", serial])
        command.extend(args)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=False,
                check=False,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired as exc:
            raise CaptureError(f"ADB 命令超时（{timeout:g} 秒）", stage="capture") from exc
        except OSError as exc:
            raise CaptureError(f"ADB 无法执行: {exc}", stage="capture") from exc
        if completed.returncode != 0:
            error = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
            raise CaptureError(f"ADB 命令失败: {error or completed.returncode}", stage="capture")
        if not isinstance(completed.stdout, bytes):
            raise CaptureError("ADB 未返回二进制输出", stage="capture")
        return completed.stdout

    def list_devices(self) -> list[dict[str, str]]:
        """列出设备状态，不启动相机、不触发快门。"""
        raw = self._run(["devices", "-l"], timeout=self.command_timeout_sec)
        devices = []
        for line in raw.decode("utf-8", errors="replace").splitlines():
            text = line.strip()
            if not text or text.startswith(("List of devices", "*", "adb server")):
                continue
            fields = text.split()
            if len(fields) >= 2:
                devices.append({"serial": fields[0], "state": fields[1], "description": text})
        return devices

    def _select_serial(self, devices: list[dict[str, str]] | None = None) -> str:
        candidates = devices if devices is not None else self.list_devices()
        if self.serial is not None:
            selected = next((item for item in candidates if item["serial"] == self.serial), None)
            if selected is None:
                raise CaptureError(f"未找到指定手机: {self.serial}", stage="capture")
            if selected["state"] != "device":
                raise CaptureError(
                    f"指定手机未授权或不可用: {self.serial}（{selected['state']}）", stage="capture"
                )
            return self.serial
        authorized = [item for item in candidates if item["state"] == "device"]
        if not authorized:
            raise CaptureError(
                "没有已授权的 ADB 手机，请检查 USB 连接并在手机上允许调试", stage="capture"
            )
        if len(authorized) != 1:
            raise CaptureError("检测到多个已授权手机，请在配置中指定 serial", stage="capture")
        return authorized[0]["serial"]

    @staticmethod
    def _decode(data: bytes, *, require_png: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        if not data:
            raise CaptureError("手机未返回图像字节", stage="capture")
        if require_png and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise CaptureError("ADB 截屏未返回有效 PNG", stage="capture")
        try:
            with Image.open(io.BytesIO(data)) as check:
                check.verify()
            with Image.open(io.BytesIO(data)) as opened:
                raw_width, raw_height = opened.size
                image_format = opened.format
                exif_orientation = int(opened.getexif().get(274, 1))
                opened.load()
                rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
            image = rgb[:, :, ::-1].copy()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            TypeError,
            SyntaxError,
            OverflowError,
            Image.DecompressionBombError,
        ) as exc:
            raise CaptureError(f"手机图像无法完整解码: {exc}", stage="capture") from exc
        height, width = image.shape[:2]
        if min(width, height) < 16:
            raise CaptureError(f"手机图像尺寸无效: {width}×{height}", stage="capture")
        if require_png and image_format != "PNG":
            raise CaptureError("ADB 截屏格式不是 PNG", stage="capture")
        raw_format = str(image_format or "unknown").lower()
        return image, {
            "raw_format": raw_format,
            "raw_suffix": {"jpeg": ".jpg", "unknown": ".bin"}.get(raw_format, "." + raw_format),
            "raw_width": raw_width,
            "raw_height": raw_height,
            "width": width,
            "height": height,
            "orientation": "landscape"
            if width > height
            else "portrait"
            if height > width
            else "square",
            "exif_orientation": exif_orientation,
        }

    @staticmethod
    def _black_stats(image: np.ndarray) -> dict[str, float | bool]:
        # 中心区域排除状态栏和快门按钮，识别受保护或不可用的纯黑预览。
        height, width = image.shape[:2]
        center = image[height // 5 : height - height // 5, width // 5 : width - width // 5]
        maximum = center.max(axis=2)
        mean = float(maximum.mean())
        percentile = float(np.percentile(maximum, 99))
        near_black_ratio = float((maximum <= 16).mean())
        return {
            "mean": round(mean, 3),
            "p99": round(percentile, 3),
            "near_black_ratio": round(near_black_ratio, 4),
            "black": mean <= 8 and (percentile <= 16 or near_black_ratio >= 0.98),
        }

    def _foreground_camera(self, serial: str, deadline: float) -> str:
        self._assert_unlocked(serial, deadline)
        data = self._run(
            ["shell", "dumpsys", "activity", "activities"],
            serial=serial,
            timeout=self._timeout(deadline),
        ).decode("utf-8", errors="replace")
        package = None
        for marker in ("topResumedActivity", "mResumedActivity", "ResumedActivity:"):
            packages = set()
            for line in data.splitlines():
                if marker not in line or "mLastResumedActivity" in line:
                    continue
                match = re.search(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)/", line)
                if match:
                    packages.add(match.group(1))
            if packages:
                if len(packages) == 1:
                    package = next(iter(packages))
                break
        expected = self.camera_shutter["camera_package"]
        if package != expected:
            raise CaptureError(
                f"手机前台不是已配置相机（期望 {expected}，实际 {package or '无法确认'}），未发送触控或快门",
                stage="capture",
            )
        return package

    def _assert_unlocked(self, serial: str, deadline: float) -> None:
        """只读核对当前 keyguard；设备支持锁屏或设置了密码并不等于已锁。"""
        policy = self._run(
            ["shell", "dumpsys", "window", "policy"],
            serial=serial,
            timeout=self._timeout(deadline),
        ).decode("utf-8", errors="replace")
        # Huawei 可能保留过期的 showingAndNotOccluded=true；优先采用
        # KeyguardStateMonitor 的 mIsShowing，不能让组合缓存覆盖当前事实。
        locked = None
        for name in ("mIsShowing", "showing", "showingAndNotOccluded"):
            values = re.findall(
                rf"(?<![\w]){name}\s*[=:]\s*(true|false)\b", policy, flags=re.IGNORECASE
            )
            if values:
                current = {value.lower() for value in values}
                if len(current) == 1:
                    locked = current.pop() == "true"
                break
        if locked is None:
            raise CaptureError("无法确认手机当前锁屏状态，请先解锁并打开原生相机", stage="capture")
        if locked:
            raise CaptureError(
                "手机已锁屏，请先解锁并打开原生相机",
                stage="capture",
                details={"phone_locked": True},
            )

    def _check_touch_size(self, width: int, height: int) -> None:
        expected = self.camera_shutter["screen_size"]
        if expected is None or [width, height] != expected:
            raise CaptureError(
                f"手机截图尺寸 {width}×{height} 与触控标定尺寸 {expected} 不一致，未发送触控",
                stage="capture",
            )

    def _read_screen(
        self, serial: str, deadline: float
    ) -> tuple[bytes, np.ndarray, dict[str, Any]]:
        data = self._run(
            ["exec-out", "screencap", "-p"],
            serial=serial,
            timeout=min(self.screenshot_timeout_sec, self._timeout(deadline)),
        )
        image, meta = self._decode(data, require_png=True)
        return data, image, meta

    def _wake_camera(self, serial: str, width: int, height: int, deadline: float) -> None:
        position = self.camera_shutter["wake_position"]
        if position is None:
            raise CaptureError("手机相机预览黑屏，未配置安全唤醒坐标", stage="capture")
        self._check_touch_size(width, height)
        self._foreground_camera(serial, deadline)
        try:
            self._run(
                ["shell", "input", "tap", str(position[0]), str(position[1])],
                serial=serial,
                timeout=self._timeout(deadline),
            )
        except CaptureError as exc:
            # 超时不能证明触控未执行，后续回退不得再发送一次唤醒。
            exc.details = {**(exc.details or {}), "wake_attempted": True}
            raise
        remaining = deadline - time.monotonic()
        delay = self.camera_shutter["wake_delay_sec"]
        if remaining <= delay:
            raise CaptureError(
                "手机相机唤醒超过采集截止时间", stage="capture", details={"wake_attempted": True}
            )
        time.sleep(delay)
        try:
            self._foreground_camera(serial, deadline)
        except CaptureError as exc:
            exc.details = {**(exc.details or {}), "wake_attempted": True}
            raise

    def _capture_preview(self, serial: str, *, allow_wake: bool = False) -> PhoneCaptureResult:
        started = time.monotonic()
        deadline = started + self.screenshot_timeout_sec
        wake_count = 0
        if self.camera_shutter["method"] == "tap":
            self._foreground_camera(serial, deadline)
        for attempt in range(2):
            try:
                data, image, meta = self._read_screen(serial, deadline)
            except CaptureError as exc:
                if wake_count:
                    exc.details = {**(exc.details or {}), "wake_attempted": True}
                raise
            height, width = image.shape[:2]
            if self.preview_screen_size is not None and (width, height) != self.preview_screen_size:
                expected_width, expected_height = self.preview_screen_size
                raise CaptureError(
                    f"手机截图尺寸 {width}×{height} 与预览标定尺寸 "
                    f"{expected_width}×{expected_height} 不一致，请恢复取景方向或重新标定",
                    stage="capture",
                    details={
                        "actual_screen_size": [width, height],
                        "expected_screen_size": list(self.preview_screen_size),
                    },
                )
            bounds = [0, 0, width, height]
            if self.preview_roi is not None:
                left, top, right, bottom = self.preview_roi
                bounds = [
                    math.floor(left * width),
                    math.floor(top * height),
                    math.ceil(right * width),
                    math.ceil(bottom * height),
                ]
                image = image[bounds[1] : bounds[3], bounds[0] : bounds[2]].copy()
            black = self._black_stats(image)
            if not black["black"]:
                break
            if allow_wake and attempt == 0 and self.camera_shutter["wake_position"] is not None:
                self._wake_camera(serial, width, height, deadline)
                wake_count = 1
            else:
                raise CaptureError(
                    "手机相机预览异常黑屏，请确认原生相机已打开",
                    stage="capture",
                    details={"wake_attempted": bool(wake_count)},
                )
        elapsed = time.monotonic() - started
        if elapsed > self.screenshot_timeout_sec:
            raise CaptureError(
                f"手机截屏采集超过截止时间（{self.screenshot_timeout_sec:g} 秒）",
                stage="capture",
                details={"wake_attempted": bool(wake_count)},
            )
        meta.update(
            serial=serial,
            source="preview_screenshot",
            preview_roi=list(self.preview_roi) if self.preview_roi is not None else None,
            preview_screen_size=list(self.preview_screen_size)
            if self.preview_screen_size is not None
            else None,
            preview_bounds_px=bounds,
            preview_width=int(image.shape[1]),
            preview_height=int(image.shape[0]),
            black_check=black,
            screenshot_elapsed_sec=round(elapsed, 4),
            shutter_count=0,
            wake_count=wake_count,
        )
        return PhoneCaptureResult(data, "preview_screenshot", image, meta)

    def _timeout(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CaptureError("等待手机新照片超时，未复用旧照片", stage="capture")
        return min(self.command_timeout_sec, remaining)

    def _list_photos(self, serial: str, deadline: float) -> set[str]:
        command = shlex.join(["find", self.photo_dir, "-maxdepth", "1", "-type", "f", "-print0"])
        output = self._run(["shell", command], serial=serial, timeout=self._timeout(deadline))
        paths = set()
        for entry in output.split(b"\x00"):
            if not entry:
                continue
            try:
                path = entry.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CaptureError("手机照片路径无法解码", stage="capture") from exc
            remote = PurePosixPath(path)
            # Huawei / MediaStore 会先写 .pending-...jpg，再原子改名为最终照片。
            # 隐藏临时文件不进入新增照片集合，避免 stat/pull 遇到改名竞态。
            if (
                not remote.name.startswith(".")
                and str(remote.parent) == self.photo_dir
                and remote.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".heic",
                    ".heif",
                }
            ):
                paths.add(path)
        return paths

    def _photo_size(self, serial: str, path: str, deadline: float) -> int:
        command = shlex.join(["stat", "-c", "%s", path])
        data = self._run(["shell", command], serial=serial, timeout=self._timeout(deadline)).strip()
        try:
            size = int(data)
        except ValueError as exc:
            raise CaptureError("手机照片大小无效", stage="capture") from exc
        if size < 0:
            raise CaptureError("手机照片大小无效", stage="capture")
        return size

    def _capture_photo(
        self, serial: str, preview_error: str, *, allow_wake: bool = True
    ) -> PhoneCaptureResult:
        started = time.monotonic()
        deadline = started + self.photo_timeout_sec
        package = self._foreground_camera(serial, deadline)
        wake_count = 0
        shutter = self.camera_shutter
        if shutter["method"] == "tap" or shutter["wake_position"] is not None:
            _, screen, _ = self._read_screen(serial, deadline)
            height, width = screen.shape[:2]
            self._check_touch_size(width, height)
            if self._black_stats(screen)["black"]:
                if not allow_wake:
                    raise CaptureError(
                        "手机相机已尝试唤醒但预览仍黑屏，未再次唤醒或发送快门", stage="capture"
                    )
                self._wake_camera(serial, width, height, deadline)
                wake_count = 1
                _, screen, _ = self._read_screen(serial, deadline)
                height, width = screen.shape[:2]
                self._check_touch_size(width, height)
                if self._black_stats(screen)["black"]:
                    raise CaptureError("手机相机唤醒后预览仍黑屏，未发送快门", stage="capture")
        baseline = self._list_photos(serial, deadline)
        self._foreground_camera(serial, deadline)
        # 只触发一次快门；失败后不会补按或重启相机。
        shutter_args = (
            ["shell", "input", "tap", *map(str, shutter["position"])]
            if shutter["method"] == "tap"
            else ["shell", "input", "keyevent", "KEYCODE_CAMERA"]
        )
        self._run(
            shutter_args,
            serial=serial,
            timeout=self._timeout(deadline),
        )
        previous: tuple[str, int] | None = None
        rejected: set[tuple[str, int]] = set()
        last_error: str | None = None
        while time.monotonic() < deadline:
            new_paths = self._list_photos(serial, deadline) - baseline
            if len(new_paths) > 1:
                raise CaptureError("快门后出现多张新照片，无法唯一确定本次作品", stage="capture")
            if new_paths:
                path = next(iter(new_paths))
                size = self._photo_size(serial, path, deadline)
                current = (path, size)
                if size > 0 and previous == current and current not in rejected:
                    with tempfile.TemporaryDirectory(prefix="amcc-phone-") as temporary:
                        local_path = Path(temporary) / "capture.bin"
                        self._run(
                            ["pull", path, str(local_path)],
                            serial=serial,
                            timeout=self._timeout(deadline),
                        )
                        try:
                            data = local_path.read_bytes()
                        except OSError as exc:
                            raise CaptureError(
                                f"无法读取拉取的手机照片: {exc}", stage="capture"
                            ) from exc
                    size_after_pull = self._photo_size(serial, path, deadline)
                    if size_after_pull == size and len(data) == size:
                        try:
                            image, meta = self._decode(data)
                        except CaptureError as exc:
                            last_error = exc.message
                            rejected.add(current)
                        else:
                            meta.update(
                                serial=serial,
                                source="camera_photo",
                                remote_photo=path,
                                remote_photo_preserved=True,
                                preview_roi=None,
                                preview_screen_size=None,
                                preview_width=int(image.shape[1]),
                                preview_height=int(image.shape[0]),
                                fallback_reason=preview_error,
                                photo_elapsed_sec=round(time.monotonic() - started, 4),
                                shutter_count=1,
                                shutter_method=shutter["method"],
                                camera_package=package,
                                wake_count=wake_count,
                            )
                            return PhoneCaptureResult(data, "camera_photo", image, meta)
                    else:
                        last_error = "手机照片拉取期间大小仍在变化"
                previous = current
            else:
                previous = None
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(self.poll_interval_sec, remaining))
        reason = f"；{last_error}" if last_error else ""
        raise CaptureError(f"等待手机新照片超时，未复用旧照片{reason}", stage="capture")

    def capture(self, *, preview_only: bool = False) -> PhoneCaptureResult:
        """截屏优先；preview_only 禁止快门，适用于预览、校准与环境检查。"""
        serial = self._select_serial()
        self._assert_unlocked(serial, time.monotonic() + self.command_timeout_sec)
        try:
            return self._capture_preview(serial, allow_wake=not preview_only)
        except CaptureError as exc:
            if preview_only or not self.fallback_photo:
                raise
            wake_attempted = bool((exc.details or {}).get("wake_attempted"))
            result = self._capture_photo(serial, exc.message, allow_wake=not wake_attempted)
            if wake_attempted:
                result.meta["wake_count"] = 1
            return result

    def capture_photo(self, *, reason: str) -> PhoneCaptureResult:
        """预览存在遮挡等不可用情况时，直接采集一次新照片，不重复截屏。"""
        if not isinstance(reason, str) or not reason.strip():
            raise CaptureError("原生照片回退必须提供非空 reason", stage="capture")
        if not self.fallback_photo:
            raise CaptureError("已禁用原生照片回退", stage="capture")
        serial = self._select_serial()
        return self._capture_photo(serial, reason.strip())

    def probe(self) -> dict[str, Any]:
        """无拍照副作用的设备和预览检查，返回可直接序列化的报告。"""
        report: dict[str, Any] = {"available": False, "adb_path": str(self.adb_path)}
        try:
            devices = self.list_devices()
            report["devices"] = devices
            serial = self._select_serial(devices)
            report["serial"] = serial
            self._assert_unlocked(serial, time.monotonic() + self.command_timeout_sec)
            result = self._capture_preview(serial)
            report.update(available=True, source=result.source, preview=result.meta)
        except CaptureError as exc:
            report["error"] = exc.message
            if (exc.details or {}).get("phone_locked"):
                report["locked"] = True
        return report
