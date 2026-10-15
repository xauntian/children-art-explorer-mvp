"""摄像头采集与照片导入。

照片导入路径先实现，用于获得可重复的输入；摄像头路径后接入。
两者都只负责「拿到原始照片字节」，预处理交给 preprocess 模块。
"""

from __future__ import annotations

import time
from pathlib import Path

from .errors import CaptureError

#: 允许的原始照片扩展名（照片导入路径）
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def import_photo(path: str | Path) -> bytes:
    """导入已有照片，返回原始字节（不做任何改动，便于保留原始照片）。"""
    p = Path(path)
    if not p.is_file():
        raise CaptureError(f"照片文件不存在: {p}")
    if p.suffix.lower() not in PHOTO_SUFFIXES:
        raise CaptureError(
            f"不支持的照片格式: {p.suffix}（支持 {', '.join(PHOTO_SUFFIXES)}）"
        )
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise CaptureError(f"照片文件无法读取: {p} ({exc})") from exc
    if not data:
        raise CaptureError(f"照片文件为空: {p}")
    return data


class CameraCapture:
    """固定俯拍摄像头采集。

    只做打开、预热、抓帧；不做画框放置检测（本阶段不要求）。
    """

    def __init__(
        self,
        device_index: int = 0,
        backend: str = "dshow",
        width: int = 1920,
        height: int = 1080,
        warmup_frames: int = 5,
        settle_ms: int = 300,
    ) -> None:
        self.device_index = int(device_index)
        self.backend = backend
        self.width = int(width)
        self.height = int(height)
        self.warmup_frames = int(warmup_frames)
        self.settle_ms = int(settle_ms)

    def _api_preference(self) -> int:
        import cv2

        mapping = {
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "any": cv2.CAP_ANY,
        }
        if self.backend not in mapping:
            raise CaptureError(
                f"未知摄像头后端: {self.backend}（可选 {', '.join(mapping)}）"
            )
        return mapping[self.backend]

    def capture_jpeg(self, jpeg_quality: int = 95) -> bytes:
        """抓取一帧并编码为 JPEG 字节。"""
        import cv2

        capture = cv2.VideoCapture(self.device_index, self._api_preference())
        if not capture.isOpened():
            capture.release()
            raise CaptureError(
                f"摄像头无法打开（device_index={self.device_index}, backend={self.backend}）"
            )
        try:
            if self.width > 0:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            if self.height > 0:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            frame = None
            for _ in range(max(1, self.warmup_frames)):
                ok, candidate = capture.read()
                if ok and candidate is not None:
                    frame = candidate
                time.sleep(max(0, self.settle_ms) / 1000.0)
            if frame is None:
                raise CaptureError("摄像头未返回有效画面，请检查设备与照明")
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )
            if not ok:
                raise CaptureError("摄像头画面 JPEG 编码失败")
            return buffer.tobytes()
        finally:
            capture.release()

    def probe(self) -> dict:
        """检查摄像头是否可用，用于环境检查与故障说明。"""
        import cv2

        try:
            capture = cv2.VideoCapture(self.device_index, self._api_preference())
        except Exception as exc:  # pragma: no cover - 依赖具体驱动
            return {"available": False, "device_index": self.device_index, "error": str(exc)}
        try:
            if not capture.isOpened():
                return {
                    "available": False,
                    "device_index": self.device_index,
                    "error": "设备无法打开",
                }
            ok, frame = capture.read()
            if not ok or frame is None:
                return {
                    "available": False,
                    "device_index": self.device_index,
                    "error": "无法读取画面",
                }
            return {
                "available": True,
                "device_index": self.device_index,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            }
        finally:
            capture.release()


def list_camera_candidates(max_index: int = 4, backend: str = "dshow") -> list[dict]:
    """探测常用摄像头索引，便于在环境检查时确定 device_index。"""
    results: list[dict] = []
    for index in range(max(1, max_index)):
        results.append(CameraCapture(device_index=index, backend=backend, warmup_frames=1).probe())
    return results
