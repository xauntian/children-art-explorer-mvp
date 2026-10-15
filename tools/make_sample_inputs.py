"""生成用于调试与测试的合成粗马克笔输入。

真实验收必须使用真实儿童作品照片（见 README 的测试章节）。
本脚本只用于在没有照片时验证流程与模型链路，不替代真实输入验收。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

PAPER = (245, 245, 245)
MARKER_COLORS = [
    (40, 40, 200),
    (200, 60, 40),
    (40, 160, 40),
    (180, 40, 160),
    (30, 140, 220),
    (60, 60, 60),
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def draw_sun_and_house(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape[:2]
    cv2.circle(image, (int(w * 0.78), int(h * 0.22)), int(min(h, w) * 0.10), (30, 170, 230), 12)
    for angle in range(0, 360, 45):
        x0 = int(w * 0.78 + math.cos(math.radians(angle)) * min(h, w) * 0.14)
        y0 = int(h * 0.22 + math.sin(math.radians(angle)) * min(h, w) * 0.14)
        x1 = int(w * 0.78 + math.cos(math.radians(angle)) * min(h, w) * 0.20)
        y1 = int(h * 0.22 + math.sin(math.radians(angle)) * min(h, w) * 0.20)
        cv2.line(image, (x0, y0), (x1, y1), (30, 170, 230), 10)

    cv2.rectangle(
        image,
        (int(w * 0.12), int(h * 0.50)),
        (int(w * 0.52), int(h * 0.84)),
        (40, 40, 200),
        12,
    )
    cv2.rectangle(
        image,
        (int(w * 0.18), int(h * 0.58)),
        (int(w * 0.30), int(h * 0.70)),
        (200, 160, 40),
        -1,
    )
    cv2.rectangle(
        image,
        (int(w * 0.36), int(h * 0.62)),
        (int(w * 0.48), int(h * 0.84)),
        (60, 60, 60),
        -1,
    )
    pts = np.array(
        [
            [int(w * 0.08), int(h * 0.50)],
            [int(w * 0.32), int(h * 0.30)],
            [int(w * 0.56), int(h * 0.50)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [pts], False, (40, 160, 40), 12)


def draw_faces(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape[:2]
    for index, cx in enumerate((0.28, 0.60)):
        cy = 0.45 + (0.06 if index else -0.04)
        radius = int(min(h, w) * 0.17)
        cv2.circle(image, (int(w * cx), int(h * cy)), radius, (60, 60, 60), 12)
        cv2.circle(
            image,
            (int(w * cx - radius * 0.32), int(h * cy - radius * 0.25)),
            max(3, radius // 9),
            (60, 60, 60),
            -1,
        )
        cv2.circle(
            image,
            (int(w * cx + radius * 0.32), int(h * cy - radius * 0.25)),
            max(3, radius // 9),
            (60, 60, 60),
            -1,
        )
        cv2.ellipse(
            image,
            (int(w * cx), int(h * cy + radius * 0.30)),
            (int(radius * 0.55), int(radius * 0.34)),
            0,
            0,
            180,
            (40, 40, 200),
            10,
        )


def draw_shapes(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape[:2]
    for index in range(5):
        cx = int(w * (0.18 + 0.16 * index))
        cy = int(h * (0.34 + 0.28 * rng.random()))
        size = int(min(h, w) * (0.09 + 0.05 * rng.random()))
        color = MARKER_COLORS[index % len(MARKER_COLORS)]
        kind = index % 3
        if kind == 0:
            cv2.rectangle(image, (cx - size, cy - size), (cx + size, cy + size), color, 12)
        elif kind == 1:
            cv2.circle(image, (cx, cy), size, color, 12)
        else:
            pts = np.array(
                [[cx, cy - size], [cx + size, cy + size], [cx - size, cy + size]],
                dtype=np.int32,
            )
            cv2.polylines(image, [pts], True, color, 12)


def draw_flower(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape[:2]
    cv2.line(
        image,
        (int(w * 0.5), int(h * 0.86)),
        (int(w * 0.5), int(h * 0.42)),
        (40, 160, 40),
        12,
    )
    cv2.ellipse(image, (int(w * 0.37), int(h * 0.66)), (40, 18), -30, 0, 360, (40, 160, 40), 10)
    cv2.ellipse(image, (int(w * 0.63), int(h * 0.74)), (40, 18), 30, 0, 360, (40, 160, 40), 10)
    for angle in range(0, 360, 60):
        x = int(w * 0.5 + math.cos(math.radians(angle)) * min(h, w) * 0.15)
        y = int(h * 0.34 + math.sin(math.radians(angle)) * min(h, w) * 0.15)
        cv2.circle(image, (x, y), int(min(h, w) * 0.09), (180, 40, 160), 12)
    cv2.circle(image, (int(w * 0.5), int(h * 0.34)), int(min(h, w) * 0.07), (30, 170, 230), -1)


def draw_boat(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape[:2]
    cv2.line(image, (0, int(h * 0.55)), (w, int(h * 0.55)), (30, 140, 220), 12)
    for index in range(4):
        y = int(h * (0.62 + 0.09 * index))
        cv2.line(image, (0, y), (w, y), (60, 160, 230), 6)
    pts = np.array(
        [
            [int(w * 0.24), int(h * 0.55)],
            [int(w * 0.70), int(h * 0.55)],
            [int(w * 0.60), int(h * 0.74)],
            [int(w * 0.34), int(h * 0.74)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [pts], True, (40, 40, 200), 14)
    cv2.line(image, (int(w * 0.47), int(h * 0.55)), (int(w * 0.47), int(h * 0.18)), (60, 60, 60), 12)
    sail = np.array(
        [
            [int(w * 0.48), int(h * 0.20)],
            [int(w * 0.70), int(h * 0.50)],
            [int(w * 0.48), int(h * 0.50)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [sail], True, (180, 40, 160), 12)


MOTIFS = {
    "sun_house": draw_sun_and_house,
    "faces": draw_faces,
    "shapes": draw_shapes,
    "flower": draw_flower,
    "boat": draw_boat,
}


def make_drawing(motif: str, size: tuple[int, int] = (900, 1200), seed: int = 0) -> np.ndarray:
    """画纸 + 粗马克笔图案（不含画框）。"""
    width, height = size
    image = np.full((height, width, 3), PAPER, dtype=np.uint8)
    # 轻微纸纹，避免完全平坦
    noise = _rng(seed).integers(-4, 5, (height, width, 1), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    MOTIFS[motif](image, _rng(seed + 1))
    return image


def wrap_in_frame(
    drawing: np.ndarray,
    frame_ratio: float = 0.10,
    background: int = 55,
    perspective: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """把画纸放进底板并模拟俯拍照片（含透视与阴影）。"""
    h, w = drawing.shape[:2]
    margin = int(min(h, w) * frame_ratio)
    canvas = np.full((h + 2 * margin, w + 2 * margin, 3), background, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = drawing

    # 画框内边界（细深色线，模拟画框或底板定位）
    cv2.rectangle(canvas, (margin - 3, margin - 3), (margin + w + 2, margin + h + 2), (25, 25, 25), 6)

    if not perspective:
        return canvas

    ch, cw = canvas.shape[:2]
    jitter = _rng(seed).integers(-int(min(ch, cw) * 0.02), int(min(ch, cw) * 0.02), (4, 2))
    src = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    dst = src + jitter.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        canvas, matrix, (cw, ch), borderMode=cv2.BORDER_REPLICATE
    )

    # 轻微不均匀照明
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    gradient = 1.0 - 0.12 * ((xx / cw) ** 2 + (yy / ch) ** 2)
    warped = np.clip(warped.astype(np.float32) * gradient[..., None], 0, 255).astype(np.uint8)
    return warped


def build_dataset(
    out_dir: Path,
    count: int,
    prefix: str,
    motifs: list[str] | None = None,
    size: tuple[int, int] = (900, 1200),
    start_seed: int = 0,
) -> list[Path]:
    motifs = motifs or list(MOTIFS)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(count):
        motif = motifs[index % len(motifs)]
        seed = start_seed + index
        drawing = make_drawing(motif, size=size, seed=seed)
        photo = wrap_in_frame(drawing, seed=seed)
        path = out_dir / f"{prefix}_{index + 1:02d}_{motif}.jpg"
        cv2.imwrite(str(path), photo, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成粗马克笔测试输入")
    parser.add_argument("--out", default="samples/synthetic", help="输出目录")
    parser.add_argument("--count", type=int, default=5, help="数量")
    parser.add_argument("--prefix", default="debug", help="文件名前缀")
    parser.add_argument("--seed", type=int, default=0, help="起始随机种子")
    args = parser.parse_args()

    paths = build_dataset(
        Path(args.out), args.count, args.prefix, start_seed=args.seed
    )
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
