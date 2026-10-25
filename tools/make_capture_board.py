"""Build the exact capture board and a clearly identified synthetic test composite.

Run from the project root with .venv/Scripts/python.exe tools/make_capture_board.py.
The demo preserves the supplied family artwork; it is not a camera or AI image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOARD_SIZE = (1200, 1600)
CELL_SIZE = 24
COLORS_RGB = ((105, 119, 128), (124, 137, 145))
MARKER_SIDE = 112
QUIET_ZONE = 16
MARGIN = 24
ZONE = (176, 220, 1024, 1340)
DEMO_SAFE_MARGIN = 24
MARKERS = (
    (0, "top_left", 96, 96),
    (1, "top_right", 1104, 96),
    (2, "bottom_right", 1104, 1504),
    (3, "bottom_left", 96, 1504),
)


def _marker_rectangles(marker: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge identical rows into exact integer vector rectangles, without tracing."""
    rectangles = []
    start = 0
    for end in range(1, marker.shape[0] + 1):
        if end < marker.shape[0] and np.array_equal(marker[start], marker[end]):
            continue
        row = marker[start] == 0
        transitions = np.diff(np.r_[False, row, False].astype(np.int8))
        for left, right in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=True):
            rectangles.append((int(left), start, int(right - left), end - start))
        start = end
    return rectangles


def _marker_spec(marker_id: int, corner: str, cx: int, cy: int) -> dict:
    left, top = cx - MARKER_SIDE // 2, cy - MARKER_SIDE // 2
    right, bottom = left + MARKER_SIDE, top + MARKER_SIDE
    return {
        "id": marker_id,
        "corner": corner,
        "center_px": [cx, cy],
        "bounds_ltrb": [left, top, right, bottom],
        "quiet_bounds_ltrb": [left - QUIET_ZONE, top - QUIET_ZONE,
                              right + QUIET_ZONE, bottom + QUIET_ZONE],
        "corners_px": [[left, top], [right, top], [right, bottom], [left, bottom]],
        "rotation_clockwise_degrees": 0,
    }


def _save_png(image: Image.Image, path: Path, description: str) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("Description", description)
    image.save(path, pnginfo=info)


def _verify_markers(image: Image.Image, dictionary) -> list[int]:
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(np.asarray(image.convert("L")))
    detected = [] if ids is None else sorted(ids.reshape(-1).astype(int).tolist())
    if detected != [0, 1, 2, 3]:
        raise RuntimeError(f"All four board markers must decode exactly once; detected {detected}")
    return detected


def make_assets(output_dir: Path, family_image: Path) -> dict:
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "generateImageMarker"):
        raise RuntimeError("This OpenCV installation does not provide cv2.aruco.generateImageMarker")
    width, height = BOARD_SIZE
    yy, xx = np.indices((height, width))
    parity = ((xx // CELL_SIZE) + (yy // CELL_SIZE)) % 2
    board = np.asarray(COLORS_RGB, dtype=np.uint8)[parity]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_specs = []
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" shape-rendering="crispEdges">',
        "<title>Capture board: keep all four ArUco markers completely visible.</title>",
        "<desc>Exact vector board. The central placement zone is unlabelled and contains only the background pattern.</desc>",
        '<defs><pattern id="checker" width="48" height="48" patternUnits="userSpaceOnUse">',
        '<rect width="48" height="48" fill="rgb(105,119,128)"/>',
        '<path d="M24 0h24v24H24zM0 24h24v24H0z" fill="rgb(124,137,145)"/>',
        "</pattern></defs>",
        f'<rect width="{width}" height="{height}" fill="url(#checker)"/>',
    ]
    for marker_id, corner, cx, cy in MARKERS:
        spec = _marker_spec(marker_id, corner, cx, cy)
        marker_specs.append(spec)
        left, top, right, bottom = spec["bounds_ltrb"]
        qleft, qtop, qright, qbottom = spec["quiet_bounds_ltrb"]
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIDE, borderBits=1)
        board[qtop:qbottom, qleft:qright] = 255
        board[top:bottom, left:right] = marker[:, :, None]
        svg.append(f'<g id="aruco-{marker_id}" fill="black">')
        svg.append(f'<rect x="{qleft}" y="{qtop}" width="{qright-qleft}" height="{qbottom-qtop}" fill="white"/>')
        for mx, my, mw, mh in _marker_rectangles(marker):
            svg.append(f'<rect x="{left+mx}" y="{top+my}" width="{mw}" height="{mh}"/>')
        svg.append("</g>")
    svg.append("</svg>")
    svg_source = "\n".join(svg) + "\n"
    ElementTree.fromstring(svg_source)
    board_image = Image.fromarray(board)

    # Rotate the complete supplied image, then contain it inside the placement zone.
    # There is no crop, segmentation, generated content, border or visible UI label.
    source_bytes = family_image.read_bytes()
    with Image.open(family_image) as opened:
        source = opened.convert("RGB")
    rotated = source.transpose(Image.Transpose.ROTATE_270)  # 90 degrees clockwise.
    zl, zt, zr, zb = ZONE
    scale = min(
        (zr - zl - DEMO_SAFE_MARGIN * 2) / rotated.width,
        (zb - zt - DEMO_SAFE_MARGIN * 2) / rotated.height,
    )
    fitted_size = (int(round(rotated.width * scale)), int(round(rotated.height * scale)))
    fitted = rotated.resize(fitted_size, Image.Resampling.LANCZOS)
    paste_left = zl + ((zr - zl) - fitted.width) // 2
    paste_top = zt + ((zb - zt) - fitted.height) // 2
    demo = board_image.copy()
    demo.paste(fitted, (paste_left, paste_top))
    board_ids = _verify_markers(board_image, dictionary)
    demo_ids = _verify_markers(demo, dictionary)

    try:
        source_name = family_image.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_name = str(family_image.resolve())
    metadata = {
        "schema_version": 1,
        "board_size_px": list(BOARD_SIZE),
        "coordinate_system": "top_left_xy_pixels; bounds_ltrb_exclusive",
        "aruco": {
            "dictionary": "DICT_4X4_50",
            "marker_side_px": MARKER_SIDE,
            "border_bits": 1,
            "quiet_zone_px": QUIET_ZONE,
            "margin_px": MARGIN,
            "marker_order": "top_left, top_right, bottom_right, bottom_left",
            "corner_order": "top_left, top_right, bottom_right, bottom_left",
            "corners_convention": "geometric exterior edges; right and bottom exclusive",
            "all_four_required": True,
            "markers": marker_specs,
        },
        "background": {
            "checker_cell_px": CELL_SIZE,
            "origin_px": [0, 0],
            "colors_rgb": [list(color) for color in COLORS_RGB],
            "parity": "(floor(x/24) + floor(y/24)) modulo 2 indexes colors_rgb",
        },
        "artwork_zone_ltrb": list(ZONE),
        "demo": {
            "file": "capture-board-fast-demo.png",
            "kind": "synthetic_test_composite",
            "is_camera_capture": False,
            "ai_modified": False,
            "source": source_name,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_size_px": list(source.size),
            "rotation_clockwise_degrees": 90,
            "restore_clockwise_degrees": 270,
            "fitting": "contain; complete original image; no crop",
            "safety_margin_px": DEMO_SAFE_MARGIN,
            "scale": scale,
            "image_size_px": list(fitted_size),
            "image_bounds_ltrb": [paste_left, paste_top, paste_left + fitted.width, paste_top + fitted.height],
        },
        "validation": {
            "board_detected_ids": board_ids,
            "demo_detected_ids": demo_ids,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capture-board-fast.svg").write_text(svg_source, encoding="utf-8")
    _save_png(board_image, output_dir / "capture-board-fast.png",
              "Capture board, 1200 x 1600. Keep all four ArUco markers and their white quiet zones visible.")
    _save_png(demo, output_dir / "capture-board-fast-demo.png",
              "DEMO: synthetic test composite of the supplied family artwork and capture board. Not a camera capture. No AI modification; complete source rotated and resized only.")
    (output_dir / "capture-board-fast.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "assets")
    parser.add_argument("--family-image", type=Path,
                        default=PROJECT_ROOT / "samples" / "children_originals" / "02_family.png")
    args = parser.parse_args()
    metadata = make_assets(args.output_dir, args.family_image)
    print(f"Board SVG/PNG/JSON: {args.output_dir.resolve() / 'capture-board-fast'}")
    print(f"DEMO (synthetic composite, no AI): {args.output_dir.resolve() / metadata['demo']['file']}")
    print("Verified all four ArUco IDs on both images: 0, 1, 2, 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
