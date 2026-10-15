"""固定区域裁切与必要的透视校正，以及模型输入准备。

预处理只做「找作品边界 → 透视校正 → 缩放到输入尺寸」，
不改变原作内容，不写入 runs/ 之外的位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import PreprocessError

#: 生成模型的 VAE 通常下采样 8 倍，宽高取 8 的倍数是安全做法
SIZE_MULTIPLE = 8


@dataclass
class PreprocessResult:
    image: np.ndarray          # BGR, 模型输入尺寸
    crop_used: str             # "detected" 或 "fallback"
    corners: list[list[float]] | None
    source_size: tuple[int, int]   # (w, h) 原照片尺寸
    output_size: tuple[int, int]   # (w, h) 模型输入尺寸


def decode_image_bytes(data: bytes, source_name: str = "<bytes>") -> np.ndarray:
    """把字节解码成 BGR 图像；无法解码或尺寸无效时抛出技术错误。"""
    if not data:
        raise PreprocessError(f"图像文件为空: {source_name}")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessError(f"图像无法解码: {source_name}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessError(f"图像通道数不是 3: {source_name}")
    height, width = image.shape[:2]
    if width < 16 or height < 16:
        raise PreprocessError(f"图像尺寸无效: {source_name} ({width}x{height})")
    return image


def load_image(path: Path) -> np.ndarray:
    """读取图像文件并按 EXIF 方向纠正后解码为 BGR。"""
    path = Path(path)
    if not path.is_file():
        raise PreprocessError(f"图像文件不存在: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreprocessError(f"图像文件无法读取: {path} ({exc})") from exc
    return decode_image_bytes(raw, str(path))


def _order_corners(points: np.ndarray) -> np.ndarray:
    """把 4 个角点排序为 左上、右上、右下、左下。"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def find_frame_corners(
    image: np.ndarray,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 0.98,
    approx_epsilon_ratio: float = 0.02,
) -> np.ndarray | None:
    """在照片中寻找画框／画纸的 4 角轮廓。

    找不到可靠轮廓时返回 None，由调用方回退到固定比例裁切。
    """
    height, width = image.shape[:2]
    total_area = float(height * width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)

    candidates: list[np.ndarray] = []
    # 1) 明亮纸张区域优先：白色画纸在固定俯拍下通常是最大亮区
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=3)
    candidates.append(bright)

    # 2) 边缘图兜底：纸面与底板／背景对比不足时使用
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    candidates.append(edges)

    for mask in candidates:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: np.ndarray | None = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            ratio = area / total_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, approx_epsilon_ratio * perimeter, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue
            if area > best_area:
                best_area = area
                best = approx
        if best is not None:
            return _order_corners(best)
    return None


def warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """按 4 角做透视校正，输出与原作长宽比一致的矩形。"""
    pts = _order_corners(corners)
    (tl, tr, br, bl) = pts
    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    target_w = int(round(max(width_a, width_b)))
    target_h = int(round(max(height_a, height_b)))
    if target_w < 16 or target_h < 16:
        raise PreprocessError("透视校正目标尺寸无效，检测到的角点可能过于靠近")
    destination = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, destination)
    return cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_LINEAR)


def trim_border(image: np.ndarray, ratio: float) -> np.ndarray:
    """按比例裁掉四周边缘（回退路径，去掉画框与背景残留）。"""
    if ratio <= 0:
        return image
    height, width = image.shape[:2]
    dx = int(round(width * ratio))
    dy = int(round(height * ratio))
    if width - 2 * dx < 16 or height - 2 * dy < 16:
        return image
    return image[dy : height - dy, dx : width - dx]


def crop_ratios_region(image: np.ndarray, ratios: tuple[float, float, float, float]) -> np.ndarray:
    """按 (left, top, right, bottom) 比例做固定区域裁切。"""
    height, width = image.shape[:2]
    left, top, right, bottom = (float(r) for r in ratios)
    x0 = int(round(width * left))
    y0 = int(round(height * top))
    x1 = int(round(width * (1.0 - right)))
    y1 = int(round(height * (1.0 - bottom)))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise PreprocessError("固定区域裁切结果过小，请检查 crop_ratios 配置")
    return image[y0:y1, x0:x1]


def resize_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """等比缩放到最长边 = max_side，并把宽高对齐到 SIZE_MULTIPLE 的倍数。"""
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(width, height)))
    new_w = max(SIZE_MULTIPLE, int(round(width * scale)))
    new_h = max(SIZE_MULTIPLE, int(round(height * scale)))
    new_w -= new_w % SIZE_MULTIPLE
    new_h -= new_h % SIZE_MULTIPLE
    if new_w == width and new_h == height:
        return image
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def preprocess(
    image: np.ndarray,
    max_input_side: int,
    frame_detect: bool = True,
    border_trim_ratio: float = 0.06,
    crop_ratios: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 0.98,
    approx_epsilon_ratio: float = 0.02,
) -> PreprocessResult:
    """完整预处理：透视校正 → 固定区域裁切 → 缩放到输入尺寸。"""
    if image is None or image.size == 0:
        raise PreprocessError("预处理输入为空图像")
    source_size = (int(image.shape[1]), int(image.shape[0]))

    corners = None
    crop_used = "fallback"
    if frame_detect:
        corners = find_frame_corners(
            image,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            approx_epsilon_ratio=approx_epsilon_ratio,
        )
        if corners is not None:
            try:
                image = warp_perspective(image, corners)
                crop_used = "detected"
            except PreprocessError:
                corners = None
                crop_used = "fallback"

    if crop_used == "fallback":
        image = crop_ratios_region(image, crop_ratios)
        image = trim_border(image, border_trim_ratio)

    resized = resize_to_max_side(image, max_input_side)
    output_size = (int(resized.shape[1]), int(resized.shape[0]))
    return PreprocessResult(
        image=resized,
        crop_used=crop_used,
        corners=None if corners is None else [[float(x), float(y)] for x, y in corners],
        source_size=source_size,
        output_size=output_size,
    )


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise PreprocessError("PNG 编码失败")
    return buffer.tobytes()


def prepare_model_input(
    source_path: Path,
    max_input_side: int,
    frame_detect: bool = True,
    border_trim_ratio: float = 0.06,
    crop_ratios: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 0.98,
    approx_epsilon_ratio: float = 0.02,
) -> tuple[PreprocessResult, bytes]:
    """从照片文件生成模型输入 PNG 字节。"""
    image = load_image(source_path)
    result = preprocess(
        image,
        max_input_side=max_input_side,
        frame_detect=frame_detect,
        border_trim_ratio=border_trim_ratio,
        crop_ratios=crop_ratios,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        approx_epsilon_ratio=approx_epsilon_ratio,
    )
    return result, encode_png(result.image)
