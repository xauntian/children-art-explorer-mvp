"""预处理与图像校验测试。

要求：保留原始照片；裁切后不包含画框外背景；文件损坏时按技术错误处理。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import PreprocessError
from app.preprocess import (
    SIZE_MULTIPLE,
    crop_ratios_region,
    decode_image_bytes,
    encode_png,
    find_frame_corners,
    load_image,
    preprocess,
    resize_to_max_side,
    trim_border,
    warp_perspective,
)


def _photo_with_frame(
    width: int = 600, height: int = 480, frame: tuple[int, int, int, int] = (100, 60, 500, 420)
) -> np.ndarray:
    """深色背景 + 明亮画纸的合成俯拍照片。"""
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    x0, y0, x1, y1 = frame
    image[y0:y1, x0:x1] = 240
    # 画纸上的粗马克笔图案
    image[y0 + 40 : y0 + 100, x0 + 40 : x0 + 160] = (0, 0, 200)
    return image


# ---------------------------------------------------------------- 解码
def test_decode_rejects_empty_bytes():
    with pytest.raises(PreprocessError, match="为空"):
        decode_image_bytes(b"")


def test_decode_rejects_corrupt_bytes():
    with pytest.raises(PreprocessError, match="无法解码"):
        decode_image_bytes(b"this is not an image at all")


def test_decode_rejects_truncated_png():
    good = encode_png(_photo_with_frame())
    with pytest.raises(PreprocessError):
        decode_image_bytes(good[: len(good) // 3])


def test_decode_rejects_tiny_image():
    import cv2

    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", tiny)
    assert ok
    with pytest.raises(PreprocessError, match="尺寸无效"):
        decode_image_bytes(buf.tobytes())


def test_load_image_missing_file(tmp_path):
    with pytest.raises(PreprocessError, match="不存在"):
        load_image(tmp_path / "nope.png")


def test_png_roundtrip_preserves_size():
    original = _photo_with_frame(320, 240)
    decoded = decode_image_bytes(encode_png(original))
    assert decoded.shape == original.shape


# ---------------------------------------------------------------- 画框检测
def test_finds_frame_corners_on_synthetic_photo():
    corners = find_frame_corners(_photo_with_frame())
    assert corners is not None, "明亮画纸应能被检测到"
    xs = sorted(corners[:, 0])
    ys = sorted(corners[:, 1])
    assert xs[0] == pytest.approx(100, abs=15)
    assert xs[-1] == pytest.approx(500, abs=15)
    assert ys[0] == pytest.approx(60, abs=15)
    assert ys[-1] == pytest.approx(420, abs=15)


def test_frame_detection_returns_none_on_flat_image():
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    assert find_frame_corners(flat) is None


def test_warp_perspective_removes_background():
    image = _photo_with_frame()
    corners = find_frame_corners(image)
    assert corners is not None
    warped = warp_perspective(image, corners)
    h, w = warped.shape[:2]
    # 校正后四角应接近画纸亮度，而不是深色背景
    for corner in (warped[2, 2], warped[2, w - 3], warped[h - 3, 2], warped[h - 3, w - 3]):
        assert corner.mean() > 180, "透视校正后不应残留画框外背景"


def test_warp_rejects_degenerate_corners():
    degenerate = np.array([[10, 10], [12, 10], [12, 12], [10, 12]], dtype=np.float32)
    with pytest.raises(PreprocessError, match="目标尺寸无效"):
        warp_perspective(np.zeros((100, 100, 3), dtype=np.uint8), degenerate)


# ---------------------------------------------------------------- 裁切与缩放
def test_crop_ratios_region_trims_requested_borders():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = crop_ratios_region(image, (0.1, 0.2, 0.1, 0.2))
    assert cropped.shape[0] == pytest.approx(60, abs=1)
    assert cropped.shape[1] == pytest.approx(160, abs=1)


def test_crop_ratios_region_rejects_too_small_result():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(PreprocessError, match="过小"):
        crop_ratios_region(image, (0.49, 0.49, 0.49, 0.49))


def test_trim_border_removes_margin():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    trimmed = trim_border(image, 0.1)
    assert trimmed.shape[:2] == (80, 80)


def test_resize_limits_long_side_and_aligns_to_multiple():
    image = np.zeros((1000, 500, 3), dtype=np.uint8)
    resized = resize_to_max_side(image, 512)
    h, w = resized.shape[:2]
    assert max(h, w) <= 512
    assert w % SIZE_MULTIPLE == 0 and h % SIZE_MULTIPLE == 0
    # 长宽比保持
    assert w / h == pytest.approx(500 / 1000, rel=0.05)


def test_resize_does_not_upscale_small_image():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    resized = resize_to_max_side(image, 768)
    assert resized.shape[:2] == (64, 64)


# ---------------------------------------------------------------- 完整预处理
def test_preprocess_keeps_aspect_ratio_and_marks_crop_method():
    result = preprocess(_photo_with_frame(), max_input_side=256, frame_detect=True)
    w, h = result.output_size
    assert result.crop_used == "detected"
    assert max(w, h) <= 256
    assert w % SIZE_MULTIPLE == 0 and h % SIZE_MULTIPLE == 0
    assert result.source_size == (600, 480)
    assert result.corners is not None


def test_preprocess_fallback_uses_fixed_crop():
    flat = np.full((400, 400, 3), 128, dtype=np.uint8)
    result = preprocess(
        flat,
        max_input_side=128,
        frame_detect=True,
        border_trim_ratio=0.1,
        crop_ratios=(0.0, 0.0, 0.0, 0.0),
    )
    assert result.crop_used == "fallback"
    assert result.corners is None
    assert result.output_size[0] <= 128


def test_preprocess_with_frame_detect_disabled():
    result = preprocess(_photo_with_frame(), max_input_side=128, frame_detect=False)
    assert result.crop_used == "fallback"


def test_preprocess_rejects_empty_image():
    with pytest.raises(PreprocessError, match="空图像"):
        preprocess(np.zeros((0, 0, 3), dtype=np.uint8), max_input_side=128)


def test_preprocess_output_is_decodable():
    result = preprocess(_photo_with_frame(), max_input_side=128)
    decoded = decode_image_bytes(encode_png(result.image))
    assert decoded.shape[:2] == (result.output_size[1], result.output_size[0])


def test_real_config_preprocess_is_valid():
    from pathlib import Path

    from app.config import load_app_config

    root = Path(__file__).resolve().parents[2]
    config = load_app_config(root)
    assert config.pre_max_input_side > 0
    assert config.pre_output_format == "png"
    assert len(config.pre_crop_ratios) == 4
