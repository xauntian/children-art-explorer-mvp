"""Two-stage checkerboard capture extraction; no camera or model side effects.

The board is a *physical pattern*, not an alpha channel.  Its calibrated grid
predicts background appearance; spatially coherent deviations define artwork.
Opaque rectangular artwork is retained as a whole, including black/white marks
and blank interiors.  Ambiguous or missing geometry raises an error instead of
silently cropping a fixed percentage of the photograph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .errors import PreprocessError


class ArtworkExtractError(PreprocessError):
    """The photograph cannot be safely separated from its checkerboard."""

    category = "artwork_extract_error"


@dataclass
class ExtractionProfile:
    # The legacy checkerboard calibration uses 32 cells along each side.
    board_expected_ratio: float | None = 1.0
    artwork_expected_ratio: float | None = None
    checker_cells: tuple[int, int] | None = (32, 32)
    board_mode: str = "full_board"  # or visible_plane: a calibrated preview ROI
    rotation_degrees: int = 0  # explicit clockwise rotation, never semantic guessing
    ignore_regions: tuple = ()  # camera UI rectangles (x0,y0,x1,y1), input coordinates
    preview_occlusion_regions: tuple = ()  # normalized input UI boxes; flag overlap, never invent pixels
    max_detection_side: int = 1000
    max_board_side: int = 1600
    min_board_area_ratio: float = 0.06
    min_checker_contrast: float = 18.0
    min_checker_agreement: float = 0.66
    foreground_tolerance: float = 36.0
    min_artwork_area_ratio: float = 0.003
    max_artwork_area_ratio: float = 0.85
    min_component_pixels: int = 18
    rectangle_fill_ratio: float = 0.84
    rectify_artwork: bool = True


@dataclass
class ArtworkExtractionResult:
    board_image: np.ndarray  # BGR, rectified board
    artwork: np.ndarray  # BGRA, cv2.imencode PNG ready
    rgba: np.ndarray  # RGBA, display/tool consumers
    model_input: np.ndarray  # BGR, alpha composited onto white
    overlay: np.ndarray  # BGR, original input coordinates
    mask: np.ndarray  # uint8 alpha in final artwork coordinates
    board_mask: np.ndarray  # uint8 alpha in rectified board coordinates
    board_corners: list[list[float]]
    artwork_corners: list[list[float]] | None  # rectified board coordinates
    meta: dict[str, Any]


def _profile(value: ExtractionProfile | dict | None) -> ExtractionProfile:
    if value is None:
        result = ExtractionProfile()
    elif isinstance(value, ExtractionProfile):
        result = ExtractionProfile(**asdict(value))
    elif isinstance(value, dict):
        try:
            result = ExtractionProfile(**value)
        except TypeError as exc:
            raise ArtworkExtractError(f"无效的提取配置: {exc}") from exc
    else:
        raise ArtworkExtractError("profile 必须是 ExtractionProfile 或配置对象")
    try:
        if result.checker_cells is not None:
            result.checker_cells = tuple(int(v) for v in result.checker_cells)
            if len(result.checker_cells) != 2 or min(result.checker_cells) < 4:
                raise ValueError("checker_cells 必须包含两个不小于 4 的整数")
            if max(result.checker_cells) > 128:
                raise ValueError("checker_cells 不能超过 128")
        if result.board_mode not in ("full_board", "visible_plane"):
            raise ValueError("board_mode 必须是 full_board 或 visible_plane")
        if result.board_mode == "full_board" and result.checker_cells is None:
            raise ValueError("完整底板模式需指定 checker_cells，裁幅预览请用 visible_plane")
        if result.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("rotation_degrees 必须是 0、90、180 或 270")
        result.ignore_regions = tuple(
            tuple(float(v) for v in region) for region in result.ignore_regions
        )
        if any(
            len(region) != 4
            or not np.all(np.isfinite(region))
            or region[2] <= region[0]
            or region[3] <= region[1]
            for region in result.ignore_regions
        ):
            raise ValueError("ignore_regions 必须包含有效的 UI 矩形坐标")
        result.preview_occlusion_regions = tuple(
            tuple(float(v) for v in region) for region in result.preview_occlusion_regions
        )
        if any(
            len(region) != 4
            or not np.all(np.isfinite(region))
            or min(region) < 0
            or max(region) > 1
            or region[2] <= region[0]
            or region[3] <= region[1]
            for region in result.preview_occlusion_regions
        ):
            raise ValueError("preview_occlusion_regions 必须是 [0,1] 范围的 UI 矩形")
        for ratio in (result.board_expected_ratio, result.artwork_expected_ratio):
            if ratio is not None and (not np.isfinite(ratio) or ratio <= 0):
                raise ValueError("预期宽高比必须是有限正数")
        if min(result.max_detection_side, result.max_board_side) < 128:
            raise ValueError("检测和底板最大边至少为 128")
        if not 0 < result.min_checker_agreement <= 1:
            raise ValueError("min_checker_agreement 必须位于 (0, 1]")
        if not 0 < result.min_artwork_area_ratio < result.max_artwork_area_ratio < 1:
            raise ValueError("画作面积范围无效")
        if result.foreground_tolerance <= 0 or result.min_checker_contrast <= 0:
            raise ValueError("颜色容差和棋盘对比度必须为正数")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtworkExtractError(f"无效的提取配置: {exc}") from exc
    return result


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        raise ArtworkExtractError("四角坐标必须是四个有限的二维点")
    if len(np.unique(pts, axis=0)) != 4:
        raise ArtworkExtractError("四角坐标包含重复点")
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
    if cv2.contourArea(pts, oriented=True) < 0:
        pts = pts[[0, 3, 2, 1]]
    if not cv2.isContourConvex(pts) or abs(cv2.contourArea(pts)) < 64:
        raise ArtworkExtractError("四角不能构成有效凸四边形")
    if min(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)) < 8:
        raise ArtworkExtractError("四角间距过小")
    return pts


def _intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    u, v = b - a, d - c
    matrix = np.column_stack((u, -v))
    if abs(float(np.linalg.det(matrix))) < 1e-5:
        return None
    return a + np.linalg.solve(matrix, c - a)[0] * u


def _quad_from_hull(contour: np.ndarray) -> np.ndarray | None:
    """Extrapolate four long sides, including black clipped checker corners."""
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    perimeter = cv2.arcLength(hull, True)
    polygon = cv2.approxPolyDP(hull, 0.006 * perimeter, True).reshape(-1, 2)
    if len(polygon) < 4:
        return None
    if len(polygon) == 4:
        candidate = polygon
    else:
        lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        chosen = sorted(np.argsort(lengths)[-4:].tolist())
        lines = [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in chosen]
        vertices = []
        for i in range(4):
            vertex = _intersect(*lines[i - 1], *lines[i])
            if vertex is None:
                return None
            vertices.append(vertex)
        candidate = np.asarray(vertices, np.float32)
    try:
        ordered = _order_quad(candidate)
    except ArtworkExtractError:
        return None
    hull_area = cv2.contourArea(hull)
    quad_area = cv2.contourArea(ordered)
    if quad_area <= 0 or not 0.70 <= hull_area / quad_area <= 1.04:
        return None
    return ordered


def _warp(image: np.ndarray, quad: np.ndarray, ratio: float | None, max_side: int):
    tl, tr, br, bl = _order_quad(quad)
    width = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))
    height = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
    if ratio is not None:
        # Preserve approximate sampling density while enforcing a known plane ratio.
        area = width * height
        width, height = float(np.sqrt(area * ratio)), float(np.sqrt(area / ratio))
    scale = min(1.0, max_side / max(width, height))
    width, height = max(16, int(round(width * scale))), max(16, int(round(height * scale)))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), destination)
    result = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return result, matrix


def _cell_samples(board: np.ndarray, cells: tuple[int, int] | None, grid=None):
    height, width = board.shape[:2]
    if grid is None:
        nx, ny = cells
        sx, sy, ox, oy = width / nx, height / ny, 0.0, 0.0
    else:
        sx, sy = grid["cell_size"]
        ox, oy = grid["origin"]
    if min(sx, sy) < 3:
        raise ArtworkExtractError("棋盘格在图像中过小，无法可靠区分背景与画作")
    values, coordinates, labels, ring = [], [], [], []
    for y in range(int(np.floor(-oy / sy)), int(np.ceil((height - oy) / sy))):
        for x in range(int(np.floor(-ox / sx)), int(np.ceil((width - ox) / sx))):
            px, py = ox + (x + 0.5) * sx, oy + (y + 0.5) * sy
            if not sx * 0.20 < px < width - sx * 0.20 or not sy * 0.20 < py < height - sy * 0.20:
                continue
            dx, dy = max(1, int(sx * 0.16)), max(1, int(sy * 0.16))
            patch = board[
                max(0, int(py) - dy) : min(height, int(py) + dy + 1),
                max(0, int(px) - dx) : min(width, int(px) + dx + 1),
            ]
            values.append(np.median(patch.reshape(-1, 3), axis=0))
            coordinates.append((px / width, py / height))
            labels.append((x + y) % 2)
            band = 0.10 if grid is not None else 0.18
            ring.append(
                px < max(sx * 0.8, width * band)
                or px > width - max(sx * 0.8, width * band)
                or py < max(sy * 0.8, height * band)
                or py > height - max(sy * 0.8, height * band)
            )
    return (
        np.array(values, np.float32),
        np.array(coordinates, np.float32),
        np.array(labels),
        np.array(ring, dtype=bool),
    )


def _basis(x, y):
    return np.stack((np.ones_like(x), x, y, x * y, x * x, y * y), axis=-1)


def _checker_model(board: np.ndarray, profile: ExtractionProfile, grid=None) -> dict:
    values, coordinates, labels, ring = _cell_samples(board, profile.checker_cells, grid)
    if any(np.sum((labels == i) & ring) < 12 for i in (0, 1)):
        raise ArtworkExtractError("底板边带可验证的交替棋盘格不足")
    medians = [np.median(values[(labels == i) & ring], axis=0) for i in (0, 1)]
    contrast = float(np.linalg.norm(medians[1] - medians[0]) / np.sqrt(3))
    if contrast < profile.min_checker_contrast:
        raise ArtworkExtractError("底板缺少可靠的交替棋盘纹理")
    design = _basis(coordinates[:, 0], coordinates[:, 1])
    coefficients = []
    residual = np.zeros(len(values), dtype=np.float32)
    for parity in (0, 1):
        available = (labels == parity) & ring
        baseline = np.max(np.abs(values - medians[parity]), axis=1)
        selected = available & (baseline < max(45.0, contrast * 0.38))
        if selected.sum() < 12:
            raise ArtworkExtractError("底板可见棋盘格不足，无法建立背景参考")
        coefficient = None
        for _ in range(3):
            coefficient = np.linalg.lstsq(design[selected], values[selected], rcond=None)[0]
            error = np.max(np.abs(values - design @ coefficient), axis=1)
            limit = max(
                15.0, min(profile.foreground_tolerance, float(np.median(error[selected])) * 3 + 8)
            )
            updated = available & (error < limit)
            if updated.sum() < 12:
                break
            selected = updated
        coefficients.append(coefficient)
        residual[labels == parity] = np.max(
            np.abs(values[labels == parity] - design[labels == parity] @ coefficient), axis=1
        )
    limit = max(profile.foreground_tolerance, 20.0)
    ring_agreement = float(np.mean(residual[ring] < limit))
    all_agreement = float(np.mean(residual < limit))
    score = 0.72 * ring_agreement + 0.28 * all_agreement
    if score < profile.min_checker_agreement or ring_agreement < 0.60:
        raise ArtworkExtractError("底板棋盘纹理与预期网格不一致，需重新标定或调整机位")
    if grid is None:
        grid = {
            "cell_size": [
                board.shape[1] / profile.checker_cells[0],
                board.shape[0] / profile.checker_cells[1],
            ],
            "origin": [0.0, 0.0],
        }
    return {
        "coefficients": coefficients,
        "contrast": contrast,
        "confidence": score,
        "grid": grid,
        "ring_agreement": ring_agreement,
        "all_agreement": all_agreement,
        "background_residual": float(np.median(residual[ring])),
    }


def _board_candidates(image: np.ndarray, profile: ExtractionProfile):
    height, width = image.shape[:2]
    scale = min(1.0, profile.max_detection_side / max(height, width))
    small = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    masks = [cv2.Canny(gray, 35, 110)]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for fraction in (0.012, 0.035, 0.065):
        size = max(3, int(round(min(gray.shape) * fraction)) | 1)
        masks.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8)))
    candidates, seen = [], set()
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < gray.size * profile.min_board_area_ratio:
                continue
            quad = _quad_from_hull(contour)
            if quad is None:
                continue
            quad = quad / scale
            if (
                np.any(quad < -3)
                or np.any(quad[:, 0] > width + 3)
                or np.any(quad[:, 1] > height + 3)
            ):
                continue
            key = tuple(np.round(quad.reshape(-1) / 5).astype(int))
            if key not in seen:
                seen.add(key)
                candidates.append(quad)
    # A tightly framed input may already be precisely the whole board.  This is
    # still checked against the grid; it is not a generic crop fallback.
    candidates.append(
        np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    )
    return candidates


def _refine_board_grid(image, board, matrix, quad, profile):
    """Refine the plane using many visible checker intersections, not art edges.

    A two-pixel board-boundary error is significant for a small camera preview.
    Grid intersection correspondences remove that phase error before masking.
    """
    height, width = board.shape[:2]
    nx, ny = profile.checker_cells
    sx, sy = width / nx, height / ny
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    response = cv2.cornerHarris(gray.astype(np.float32), 3, 3, 0.05)
    observed, target = [], []
    for y in range(1, ny):
        for x in range(1, nx):
            if 0.19 * nx < x < 0.81 * nx and 0.19 * ny < y < 0.81 * ny:
                continue
            px, py = x * sx, y * sy
            dx, dy = max(2, int(sx * 0.30)), max(2, int(sy * 0.30))
            x0, y0 = max(4, int(px) - dx), max(4, int(py) - dy)
            x1, y1 = min(width - 4, int(px) + dx + 1), min(height - 4, int(py) + dy + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            local = response[y0:y1, x0:x1]
            oy, ox = np.unravel_index(np.argmax(local), local.shape)
            if local[oy, ox] <= 0:
                continue
            point = np.array([[[float(x0 + ox), float(y0 + oy)]]], np.float32)
            cv2.cornerSubPix(
                gray,
                point,
                (3, 3),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03),
            )
            qx, qy = point[0, 0]
            delta = max(2, int(min(sx, sy) * 0.23))
            ix, iy = int(round(qx)), int(round(qy))
            if min(ix, iy) < delta or ix + delta >= width or iy + delta >= height:
                continue
            samples = [
                float(gray[iy - delta, ix - delta]),
                float(gray[iy - delta, ix + delta]),
                float(gray[iy + delta, ix + delta]),
                float(gray[iy + delta, ix - delta]),
            ]
            diagonal_contrast = abs((samples[0] + samples[2] - samples[1] - samples[3]) / 2)
            if diagonal_contrast < profile.min_checker_contrast:
                continue
            if max(abs(samples[0] - samples[2]), abs(samples[1] - samples[3])) > max(
                30, diagonal_contrast * 0.40
            ):
                continue
            observed.append((qx, qy))
            target.append((px, py))
    if len(observed) < 24:
        return quad, board, matrix
    correction, inliers = cv2.findHomography(
        np.array(observed, np.float32),
        np.array(target, np.float32),
        cv2.RANSAC,
        max(1.0, min(sx, sy) * 0.08),
    )
    if correction is None or np.mean(inliers) < 0.70:
        return quad, board, matrix
    refined_matrix = correction @ matrix
    destination = np.array(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]], np.float32
    )
    refined_quad = cv2.perspectiveTransform(destination, np.linalg.inv(refined_matrix))[0]
    if np.max(np.linalg.norm(refined_quad - quad, axis=1)) > min(sx, sy) * 0.65:
        return quad, board, matrix
    refined_board = cv2.warpPerspective(
        image, refined_matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    return refined_quad, refined_board, refined_matrix


def _edge_period(gray: np.ndarray, vertical: bool):
    """Estimate cell pitch/phase from a narrow, visible checker edge band."""
    band = (
        gray[:, : max(12, int(gray.shape[1] * 0.035))].T
        if vertical
        else gray[: max(12, int(gray.shape[0] * 0.035))]
    )
    signal = np.median(band, axis=0).astype(np.float32)
    # Photographing a monitor introduces high-frequency moire that is not a
    # checker boundary. Smooth the 1-D evidence before estimating its period.
    signal = cv2.GaussianBlur(signal[None], (9, 1), 0)[0]
    gradient = np.abs(np.diff(signal))
    threshold = max(2.5, float(np.max(gradient)) * 0.25)
    indices = np.flatnonzero(gradient > threshold)
    if len(indices) < 6:
        raise ArtworkExtractError("预览边带没有足够的棋盘周期，请保留可见棋盘边缘")
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1)
    peaks = np.array(
        [np.average(group, weights=gradient[group]) for group in groups if len(group)], np.float32
    )
    if len(peaks) < 5:
        raise ArtworkExtractError("预览边带无法可靠估计棋盘格大小")
    intervals = np.diff(peaks)
    pitch = float(np.median(intervals))
    if pitch < 6 or pitch > len(signal) / 4:
        raise ArtworkExtractError("可见棋盘周期尺寸无效")
    labels = np.r_[0, np.cumsum(np.maximum(1, np.round(intervals / pitch)).astype(int))]
    pitch, origin = np.polyfit(labels, peaks, 1)
    error = np.median(np.abs(peaks - (origin + labels * pitch)))
    if error > pitch * 0.18:
        raise ArtworkExtractError("预览边带不是稳定的周期棋盘纹理")
    return float(pitch), float(origin)


def _locate_visible_plane(image, profile, explicit):
    """Rectify a cropped portion of a checker plane, without assuming its extent.

    Only the camera preview ROI is supplied here. The board's physical outer
    corners and the original SVG cell count may be outside the photograph.
    """
    height, width = image.shape[:2]
    quad = _order_quad(
        np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
        if explicit is None
        else np.asarray(explicit, np.float32)
    )
    if np.any(quad < 0) or np.any(quad[:, 0] > width - 1) or np.any(quad[:, 1] > height - 1):
        raise ArtworkExtractError("预览平面标定四角超出当前输入")
    base, initial = _warp(image, quad, None, profile.max_board_side)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    sx, ox = _edge_period(gray, False)
    sy, oy = _edge_period(gray, True)
    cell = float(np.sqrt(sx * sy))
    border = np.zeros(gray.shape, np.uint8)
    bx, by = int(gray.shape[1] * 0.18), int(gray.shape[0] * 0.16)
    border[:by] = border[-by:] = 255
    border[:, :bx] = border[:, -bx:] = 255
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=2400, qualityLevel=0.015, minDistance=cell * 0.32, mask=border, blockSize=5
    )
    if corners is None:
        raise ArtworkExtractError("预览平面缺少可验证的棋盘交点")
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    cv2.cornerSubPix(
        gray,
        corners,
        (4, 4),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, 0.03),
    )
    matches = {}
    delta = max(2, int(cell * 0.21))
    appearance_radius = max(1, int(round(cell * 0.07)))
    for px, py in corners.reshape(-1, 2):
        ix, iy = int(round(px)), int(round(py))
        margin = delta + appearance_radius
        if min(ix, iy) < margin or ix + margin >= gray.shape[1] or iy + margin >= gray.shape[0]:
            continue
        gx, gy = int(round((px - ox) / sx)), int(round((py - oy) / sy))
        residual = float(np.hypot((px - ox) / sx - gx, (py - oy) / sy - gy))
        if residual > 0.42:
            continue
        # Four single pixels can accidentally form a saddle on crayon texture
        # or screen moire. Verify the appearance of a small area in each real
        # checker quadrant, independently of the rounded lattice coordinates.
        values = []
        for dx, dy in ((-delta, -delta), (delta, -delta), (delta, delta), (-delta, delta)):
            qx, qy = ix + dx, iy + dy
            patch = gray[qy - appearance_radius : qy + appearance_radius + 1,
                         qx - appearance_radius : qx + appearance_radius + 1]
            values.append(float(np.median(patch)))
        contrast = abs((values[0] + values[2] - values[1] - values[3]) / 2)
        if contrast < profile.min_checker_contrast or max(
            abs(values[0] - values[2]), abs(values[1] - values[3])
        ) > max(30, contrast * 0.4):
            continue
        if (gx, gy) not in matches or residual < matches[gx, gy][0]:
            matches[gx, gy] = residual, (float(px), float(py))
    if len(matches) < 18:
        raise ArtworkExtractError("预览棋盘交点不足或被遮挡，无法校正可见平面")
    observed = np.array([value[1] for value in matches.values()], np.float32)
    target = np.array([(gx * cell, gy * cell) for gx, gy in matches], np.float32)
    correction, inliers = cv2.findHomography(observed, target, cv2.RANSAC, max(1.5, cell * 0.08))
    if correction is None or np.mean(inliers) < 0.65:
        raise ArtworkExtractError("棋盘交点无法组成可靠的同一平面")
    source_quad = np.array(
        [
            [
                [0, 0],
                [base.shape[1] - 1, 0],
                [base.shape[1] - 1, base.shape[0] - 1],
                [0, base.shape[0] - 1],
            ]
        ],
        np.float32,
    )
    projected = cv2.perspectiveTransform(source_quad, correction)[0]
    minimum, maximum = projected.min(axis=0), projected.max(axis=0)
    span = maximum - minimum
    scale = min(1.0, profile.max_board_side / max(span))
    shift = np.array(
        [[scale, 0, -minimum[0] * scale], [0, scale, -minimum[1] * scale], [0, 0, 1]], np.float64
    )
    matrix = shift @ correction @ initial
    size = tuple(max(32, int(round(value * scale))) for value in span)
    board = cv2.warpPerspective(image, matrix, size)
    grid = {
        "origin": (-minimum * scale).astype(float).tolist(),
        "cell_size": [float(cell * scale)] * 2,
        "visible_intersections": int(np.count_nonzero(inliers)),
        "candidate_correspondences": len(matches),
        "inlier_ratio": float(np.mean(inliers)),
        "initial_cell_size": [sx, sy],
    }
    return quad, board, matrix, _checker_model(board, profile, grid), "visible_plane_grid"


def _locate_board(image: np.ndarray, profile: ExtractionProfile, explicit):
    if profile.board_mode == "visible_plane":
        return _locate_visible_plane(image, profile, explicit)
    height, width = image.shape[:2]
    if explicit is not None:
        quad = _order_quad(np.asarray(explicit, np.float32))
        if np.any(quad < 0) or np.any(quad[:, 0] > width - 1) or np.any(quad[:, 1] > height - 1):
            raise ArtworkExtractError("标定底板四角超出当前图像；不同采集来源必须分别标定")
        board, matrix = _warp(image, quad, profile.board_expected_ratio, profile.max_board_side)
        quad, board, matrix = _refine_board_grid(image, board, matrix, quad, profile)
        return quad, board, matrix, _checker_model(board, profile), "profile_quad"
    ranked = []
    for quad in _board_candidates(image, profile):
        try:
            board, matrix = _warp(
                image, quad, profile.board_expected_ratio, min(800, profile.max_board_side)
            )
            model = _checker_model(board, profile)
            ranked.append((model["confidence"], cv2.contourArea(quad), quad))
        except (ArtworkExtractError, cv2.error, np.linalg.LinAlgError):
            continue
    if not ranked:
        raise ArtworkExtractError("未找到完整且可验证的棋盘底板，请调整取景或标定底板四角")
    _, _, quad = max(ranked, key=lambda entry: (entry[0], entry[1]))
    board, matrix = _warp(image, quad, profile.board_expected_ratio, profile.max_board_side)
    quad, board, matrix = _refine_board_grid(image, board, matrix, quad, profile)
    return quad, board, matrix, _checker_model(board, profile), "detected"


def _paper_quad_from_evidence(board, raw, cell, ignored):
    """Fit four long paper/checker transitions, excluding camera UI evidence."""
    height, width = raw.shape
    ys, xs = np.nonzero(raw)
    if len(xs) < 100:
        return None
    bounds = [
        float(np.quantile(xs, 0.015)),
        float(np.quantile(ys, 0.015)),
        float(np.quantile(xs, 0.985)),
        float(np.quantile(ys, 0.985)),
    ]
    gray = cv2.GaussianBlur(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 35, 100)
    if ignored is not None:
        edges[ignored > 0] = 0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        max(45, int(min(width, height) * 0.12)),
        minLineLength=min(width, height) * 0.30,
        maxLineGap=cell * 1.8,
    )
    if lines is None:
        return None
    evidence = cv2.blur(raw.astype(np.float32) / 255, (max(3, int(cell * 0.38)) | 1,) * 2)
    sides = [[], [], [], []]  # left, top, right, bottom
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        dx, dy = float(x1 - x0), float(y1 - y0)
        if abs(dx) < abs(dy) * 0.18:
            for side in (0, 2):
                yy = np.linspace(max(0, bounds[1]), min(height - 1, bounds[3]), 180)
                xx = x0 + (yy - y0) * dx / dy
                mid = float(np.median(xx))
                if abs(mid - bounds[side]) > cell * 2.5:
                    continue
                direction = 1 if side == 0 else -1
                inside_x, inside_y = xx + direction * cell * 0.38, yy
                outside_x, outside_y = xx - direction * cell * 0.38, yy
                sides[side].append(
                    (
                        xx,
                        yy,
                        inside_x,
                        inside_y,
                        outside_x,
                        outside_y,
                        np.array([x0, y0], np.float32),
                        np.array([x1, y1], np.float32),
                    )
                )
        elif abs(dy) < abs(dx) * 0.18:
            for side in (1, 3):
                xx = np.linspace(max(0, bounds[0]), min(width - 1, bounds[2]), 180)
                yy = y0 + (xx - x0) * dy / dx
                mid = float(np.median(yy))
                if abs(mid - bounds[side]) > cell * 2.5:
                    continue
                direction = 1 if side == 1 else -1
                inside_x, inside_y = xx, yy + direction * cell * 0.38
                outside_x, outside_y = xx, yy - direction * cell * 0.38
                sides[side].append(
                    (
                        xx,
                        yy,
                        inside_x,
                        inside_y,
                        outside_x,
                        outside_y,
                        np.array([x0, y0], np.float32),
                        np.array([x1, y1], np.float32),
                    )
                )
    selected = []
    for candidates in sides:
        ranked = []
        for xx, yy, ix, iy, ex, ey, start, end in candidates:
            valid = (np.minimum.reduce([xx, ix, ex]) >= 0) & (
                np.maximum.reduce([xx, ix, ex]) < width
            )
            valid &= (np.minimum.reduce([yy, iy, ey]) >= 0) & (
                np.maximum.reduce([yy, iy, ey]) < height
            )
            positions = [np.round(value).astype(int) for value in (xx, yy, ix, iy, ex, ey)]
            px, py, pix, piy, pex, pey = positions
            valid &= (np.maximum.reduce([px, pix, pex]) < width) & (
                np.maximum.reduce([py, piy, pey]) < height
            )
            if ignored is not None:
                safe = np.flatnonzero(valid)
                valid[safe] &= (
                    (ignored[py[safe], px[safe]] == 0)
                    & (ignored[piy[safe], pix[safe]] == 0)
                    & (ignored[pey[safe], pex[safe]] == 0)
                )
            if np.mean(valid) < 0.14:
                continue
            score = float(
                np.mean(evidence[piy[valid], pix[valid]] - evidence[pey[valid], pex[valid]])
            )
            if score > 0.10:
                ranked.append((score, start, end))
        if not ranked:
            return None
        _, start, end = max(ranked, key=lambda item: item[0])
        selected.append((start, end))
    vertices = [
        _intersect(*selected[0], *selected[1]),
        _intersect(*selected[1], *selected[2]),
        _intersect(*selected[2], *selected[3]),
        _intersect(*selected[3], *selected[0]),
    ]
    if any(value is None for value in vertices):
        return None
    try:
        quad = _order_quad(np.array(vertices, np.float32))
    except ArtworkExtractError:
        return None
    if np.any(quad < 0) or np.any(quad[:, 0] >= width) or np.any(quad[:, 1] >= height):
        return None
    return quad


def _refine_paper_boundaries(candidate, board, predicted, parity, model, ignored):
    """Fit the *outer* paper transition, rather than an internal drawn line.

    Black checker squares provide unambiguous contrast to a light paper support.
    Opposite checker rows reveal complementary portions of the same paper edge;
    an outer consensus line joins those portions without trimming a margin.
    """
    height, width = board.shape[:2]
    cell = min(model["grid"]["cell_size"])
    actual_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32)
    expected_gray = cv2.cvtColor(predicted.astype(np.float32), cv2.COLOR_BGR2GRAY)
    dark = int(np.argmin([np.median(expected_gray[parity == i]) for i in (0, 1)]))
    positive_channel_change = np.max(board.astype(np.float32) - predicted, axis=2)
    bright_on_black = (
        ((actual_gray - expected_gray) > model["contrast"] * 0.30)
        | (positive_channel_change > model["contrast"] * 0.24)
    ) & (parity == dark)
    if ignored is not None:
        bright_on_black[ignored > 0] = False
    endpoints = [
        (candidate[0], candidate[3]),
        (candidate[0], candidate[1]),
        (candidate[1], candidate[2]),
        (candidate[3], candidate[2]),
    ]
    lines = []
    for side, (a, b) in enumerate(endpoints):
        vertical = side in (0, 2)
        normal_axis, tangent_axis = (0, 1) if vertical else (1, 0)
        slope = float((b[normal_axis] - a[normal_axis]) / (b[tangent_axis] - a[tangent_axis]))
        intercept = float(a[normal_axis] - slope * a[tangent_axis])
        low, high = sorted((float(a[tangent_axis]), float(b[tangent_axis])))
        tangents = np.linspace(
            low + cell * 0.20, high - cell * 0.20, max(120, int((high - low) / 4))
        )
        points = []
        for tangent in tangents:
            estimate = slope * tangent + intercept
            normal = np.arange(
                max(0, int(estimate - cell)),
                min(width if vertical else height, int(estimate + cell) + 1),
            )
            if not len(normal):
                continue
            t = int(round(tangent))
            if t < 0 or t >= (height if vertical else width):
                continue
            evidence = bright_on_black[t, normal] if vertical else bright_on_black[normal, t]
            padded = np.r_[False, evidence, False].astype(int)
            starts, ends = (
                np.flatnonzero(np.diff(padded) == 1),
                np.flatnonzero(np.diff(padded) == -1),
            )
            valid = np.flatnonzero(ends - starts >= max(4, cell * 0.18))
            if not len(valid):
                continue
            index = valid[0] if side in (0, 1) else valid[-1]
            edge = (
                float(normal[starts[index]]) if side in (0, 1) else float(normal[ends[index] - 1])
            )
            points.append((tangent, edge))
        if len(points) < 24:
            lines.append((a, b))
            continue
        points = np.array(points)
        centre_tangent = (low + high) / 2
        initial_position = slope * centre_tangent + intercept
        candidates = []
        for trial_slope in np.linspace(slope - 0.075, slope + 0.075, 91):
            intercepts = points[:, 1] - trial_slope * points[:, 0]
            origin = float(np.floor(intercepts.min()))
            bins = np.floor(intercepts - origin).astype(int)
            density = np.convolve(np.bincount(bins), np.ones(5), mode="full")
            for peak in np.argsort(density)[-3:]:
                trial_intercept = origin + peak - 2
                members = np.abs(intercepts - trial_intercept) <= 2.6
                if np.count_nonzero(members) < max(18, len(points) * 0.16):
                    continue
                if np.ptp(points[members, 0]) < (high - low) * 0.40:
                    continue
                position = trial_slope * centre_tangent + float(np.median(intercepts[members]))
                if abs(position - initial_position) > cell * 0.75:
                    continue
                candidates.append((position, members))
        if not candidates:
            lines.append((a, b))
            continue
        _, members = (
            min(candidates, key=lambda entry: entry[0])
            if side in (0, 1)
            else max(candidates, key=lambda entry: entry[0])
        )
        fitted_slope, fitted_intercept = np.polyfit(points[members, 0], points[members, 1], 1)
        # Choose the final boundary by evidence along the *entire* side, rather
        # than extrapolating a line supported by just one checker row. A missed
        # mark costs more than including an uncertain pixel; matched white paper
        # is neutral, while confirmed black checker background is negative.
        radius = int(cell * 0.85)
        offsets = np.arange(-radius, radius + 1)
        normal_positions = (slope * tangents[:, None] + intercept + offsets).round().astype(int)
        tangent_positions = np.broadcast_to(
            np.round(tangents).astype(int)[:, None], normal_positions.shape
        )
        panel_x, panel_y = (
            (normal_positions, tangent_positions)
            if vertical
            else (tangent_positions, normal_positions)
        )
        valid = (panel_x >= 0) & (panel_x < width) & (panel_y >= 0) & (panel_y < height)
        safe_x, safe_y = np.clip(panel_x, 0, width - 1), np.clip(panel_y, 0, height - 1)
        differences = np.max(
            np.abs(board[safe_y, safe_x].astype(np.float32) - predicted[safe_y, safe_x]), axis=2
        )
        evidence = np.where(
            differences > 36,
            1.6,
            np.where((parity[safe_y, safe_x] == dark) & (differences < 25), -1.0, 0.0),
        )
        if ignored is not None:
            valid &= ignored[safe_y, safe_x] == 0
        evidence[~valid] = 0
        cumulative = np.cumsum(evidence, axis=1)
        totals = cumulative[:, -1]
        best = None
        row_indices = np.arange(len(tangents))
        midpoint = (low + high) / 2
        for trial_slope in np.linspace(slope - 0.08, slope + 0.08, 81):
            for displacement in np.arange(-cell * 0.65, cell * 0.65 + 1, 1.0):
                trial_intercept = intercept + displacement - (trial_slope - slope) * midpoint
                positions = np.round(
                    (trial_slope - slope) * tangents + trial_intercept - intercept + radius
                ).astype(int)
                if np.any(positions < 0) or np.any(positions >= len(offsets)):
                    continue
                sums = cumulative[row_indices, positions]
                score = float(
                    np.sum(totals - 2 * sums) if side in (0, 1) else np.sum(2 * sums - totals)
                )
                if best is None or score > best[0]:
                    best = (score, trial_slope, trial_intercept)
        if best is not None:
            _, fitted_slope, fitted_intercept = best
        normals = fitted_slope * np.array([low, high]) + fitted_intercept
        pair = np.column_stack(
            (normals, [low, high]) if vertical else ([low, high], normals)
        ).astype(np.float32)
        lines.append((pair[0], pair[1]))
    vertices = [
        _intersect(*lines[0], *lines[1]),
        _intersect(*lines[1], *lines[2]),
        _intersect(*lines[2], *lines[3]),
        _intersect(*lines[3], *lines[0]),
    ]
    if any(value is None for value in vertices):
        return candidate
    try:
        refined = _order_quad(np.array(vertices, np.float32))
    except ArtworkExtractError:
        return candidate
    if np.any(refined < 0) or np.any(refined[:, 0] >= width) or np.any(refined[:, 1] >= height):
        return candidate
    return refined


def _paper_seed_from_projection(board, predicted, parity, model, ignored):
    """Locate four foreground/background change points when a Hough side is weak.

    This is measured photometric evidence, not a hull of all bright pixels. It
    seeds the same four-side refinement and must pass the residual checker gate.
    """
    height, width = board.shape[:2]
    gray = cv2.cvtColor(predicted.astype(np.float32), cv2.COLOR_BGR2GRAY)
    dark = int(np.argmin([np.median(gray[parity == i]) for i in (0, 1)]))
    distance = np.max(np.abs(board.astype(np.float32) - predicted), axis=2)
    evidence = np.where(distance > 36, 1.6, np.where((parity == dark) & (distance < 25), -1.0, 0.0))
    if ignored is not None:
        evidence[ignored > 0] = 0
    signals = [
        np.mean(evidence[int(height * 0.22) : int(height * 0.78)], axis=0),
        np.mean(evidence[:, int(width * 0.22) : int(width * 0.78)], axis=1),
    ]
    boundaries = []
    for signal in signals:
        middle = len(signal) // 2
        # Partial edge cells / optical padding may have a positive DC bias.
        # Locate the strongest measured transition, not the global cumulative
        # maximum, which would otherwise drift to the photograph's outer frame.
        smoothed = cv2.GaussianBlur(
            signal[None].astype(np.float32), (0, 0), min(model["grid"]["cell_size"]) * 0.30
        )[0]
        gradient = np.gradient(smoothed)
        start = int(np.argmax(gradient[:middle]))
        end = int(np.argmin(gradient[middle:])) + middle
        if end - start < len(signal) * 0.20 or start <= 1 or end >= len(signal) - 2:
            raise ArtworkExtractError("没有四侧可验证的纸张与棋盘分界，拒绝使用底板外框替代画作")
        if np.mean(signal[start + 3 : end - 3]) < 0.12:
            raise ArtworkExtractError("纸张区域缺少可靠的非棋盘证据")
        boundaries.append((start, end))
    (left, right), (top, bottom) = boundaries
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)


def _checker_residual_gate(board, mask, model, profile):
    """Find whole retained checker cells, including a single-cell-wide rim."""
    height, width = mask.shape
    sx, sy = model["grid"]["cell_size"]
    ox, oy = model["grid"]["origin"]
    gx_values = range(int(np.ceil(-ox / sx)), int(np.floor((width - ox) / sx)))
    gy_values = range(int(np.ceil(-oy / sy)), int(np.floor((height - oy) / sy)))
    matches = np.zeros((len(gy_values), len(gx_values)), bool)
    padded_support = cv2.copyMakeBorder(
        (mask > 200).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    retained_depth = cv2.distanceTransform(padded_support, cv2.DIST_L2, 3)[1:-1, 1:-1]
    for yi, gy in enumerate(gy_values):
        for xi, gx in enumerate(gx_values):
            x0, x1 = int(ox + (gx + 0.25) * sx), int(ox + (gx + 0.75) * sx)
            y0, y1 = int(oy + (gy + 0.25) * sy), int(oy + (gy + 0.75) * sy)
            alpha = mask[y0:y1, x0:x1]
            if not alpha.size or np.mean(alpha > 200) < 0.90:
                continue
            if np.median(retained_depth[y0:y1, x0:x1]) > min(sx, sy) * 0.90:
                continue  # interior black/white artwork is not an exterior rim
            actual = board[y0:y1, x0:x1].astype(np.float32)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            predicted = _basis(xx / width, yy / height) @ model["coefficients"][(gx + gy) % 2]
            if (
                np.mean(
                    np.max(np.abs(actual - predicted), axis=2) < profile.foreground_tolerance * 0.80
                )
                > 0.90
            ):
                matches[yi, xi] = True
    longest = 0
    for sequence in list(matches) + list(matches.T):
        run = 0
        for matched in sequence:
            run = run + 1 if matched else 0
            longest = max(longest, run)
    return {
        "checker_residual_cells": int(np.count_nonzero(matches)),
        "checker_residual_max_run": int(longest),
        "checker_residual_detected": longest >= 3,
    }


def _matte_support_edges(mask, verified_background, quad, cell, max_side):
    """Interpolate the observed paper boundary across alternating white checks.

    Dark background fragments expose the same paper side at alternate cells.
    Their inner edges constrain an alpha boundary; this also removes intervening
    white background, while the opaque interior remains untouched.
    """
    local, matrix = _warp(verified_background, quad, None, max_side)
    height, width = local.shape
    limit = max(3, int(np.ceil(cell * 0.42)))
    alpha = np.full((height, width), 255, np.uint8)
    refined_sides, maximum_inset = [], 0
    for side in range(4):
        vertical = side in (0, 2)
        tangent_size = height if vertical else width
        normal_size = width if vertical else height
        positions, boundaries = [], []
        for tangent in range(2, tangent_size - 2):
            strip = local[tangent, :] if vertical else local[:, tangent]
            if side in (0, 1):
                indices = np.flatnonzero(strip[:limit] > 180)
                if len(indices) < 2:
                    continue
                boundary = int(indices.max()) + 1
            else:
                indices = np.flatnonzero(strip[-limit:] > 180)
                if len(indices) < 2:
                    continue
                boundary = int(indices.min()) + normal_size - limit
            positions.append(tangent)
            boundaries.append(boundary)
        if len(positions) < max(24, tangent_size * 0.18):
            continue
        positions, boundaries = np.asarray(positions), np.asarray(boundaries)
        if np.ptp(positions) < tangent_size * 0.50:
            continue
        normalized = positions / tangent_size
        selected = np.ones(len(positions), bool)
        for _ in range(3):
            polynomial = np.polyfit(normalized[selected], boundaries[selected], 2)
            residual = boundaries - np.polyval(polynomial, normalized)
            updated = np.abs(residual) < max(2.5, cell * 0.035)
            if np.count_nonzero(updated) < len(positions) * 0.60:
                break
            selected = updated
        curve = np.polyval(polynomial, np.arange(tangent_size) / tangent_size)
        # Keep a subpixel conservative offset, preserving edge marks. Any narrow
        # remaining dark antialias is handled by the verified component mask.
        curve += -0.5 if side in (0, 1) else 0.5
        if side in (0, 1):
            curve = np.clip(curve, 0, limit)
            inset = float(np.max(curve))
        else:
            curve = np.clip(curve, normal_size - limit, normal_size - 1)
            inset = float(np.max(normal_size - 1 - curve))
        for tangent, boundary in enumerate(np.round(curve).astype(int)):
            if vertical:
                if side == 0:
                    alpha[tangent, :boundary] = 0
                else:
                    alpha[tangent, boundary + 1 :] = 0
            elif side == 1:
                alpha[:boundary, tangent] = 0
            else:
                alpha[boundary + 1 :, tangent] = 0
        refined_sides.append(side)
        maximum_inset = max(maximum_inset, inset)
    corrected = cv2.warpPerspective(
        alpha, np.linalg.inv(matrix), (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST
    )
    return cv2.bitwise_and(mask, corrected), refined_sides, float(maximum_inset)


def _foreground(board: np.ndarray, model: dict, profile: ExtractionProfile, ignored_mask=None):
    height, width = board.shape[:2]
    sx, sy = model["grid"]["cell_size"]
    ox, oy = model["grid"]["origin"]
    ys, xs = np.mgrid[:height, :width].astype(np.float32)
    cells_x, cells_y = (xs - ox) / sx, (ys - oy) / sy
    parity = (np.floor(cells_x).astype(int) + np.floor(cells_y).astype(int)) % 2
    design = _basis(xs / width, ys / height)
    predicted = np.zeros_like(board, dtype=np.float32)
    for i in (0, 1):
        predicted[parity == i] = (design @ model["coefficients"][i])[parity == i]
    distance = np.max(np.abs(board.astype(np.float32) - predicted), axis=2)
    tolerance = max(profile.foreground_tolerance, min(65.0, model["background_residual"] * 3 + 12))
    raw = (distance > tolerance).astype(np.uint8) * 255
    # Antialiased grid edges are neither square colour; ignore only these narrow
    # grid strips before spatial reconstruction, never key all black/white pixels.
    grid_distance = np.minimum.reduce((cells_x % 1, 1 - cells_x % 1, cells_y % 1, 1 - cells_y % 1))
    raw[grid_distance < 0.10] = 0
    cell = min(sx, sy)
    size = max(3, int(round(cell * 1.25)) | 1)
    coherent = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        np.ones((size, size), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(coherent, 8)
    cleaned = np.zeros_like(raw)
    min_pixels = max(profile.min_component_pixels, int(width * height * 0.00012))
    components = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < min_pixels:
            continue
        part = (labels == index).astype(np.uint8) * 255
        components.append(part)
        cleaned |= part
    if not components:
        raise ArtworkExtractError("底板内未检测到可分离的画作，不能把空棋盘送入生成")
    fraction = float(np.count_nonzero(cleaned) / cleaned.size)
    if fraction < profile.min_artwork_area_ratio:
        raise ArtworkExtractError("画作区域过小或不可靠，请靠近取景并保持画作完整")
    if fraction > profile.max_artwork_area_ratio:
        raise ArtworkExtractError("画作占比或背景分离异常，需保留可见棋盘边缘并重新标定")
    boundary = max(2, int(round(cell * 0.25)))
    if profile.board_mode != "visible_plane" and (
        np.any(cleaned[:boundary])
        or np.any(cleaned[-boundary:])
        or np.any(cleaned[:, :boundary])
        or np.any(cleaned[:, -boundary:])
    ):
        raise ArtworkExtractError("画作或未分离背景接触底板边缘，请保持四周棋盘可见")
    artwork_quad = None
    visible_internal_checker = None
    edge_checker_removed_pixels = 0
    partial_checker_residual_pixels = 0
    partial_checker_residual_depth = 0.0
    edge_matte_sides = []
    edge_matte_max_inset = 0.0
    # A single opaque support may contain many disconnected strokes.  Retaining
    # its complete quadrilateral protects all internal colours and white space.
    if profile.rectify_artwork:
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Partial checker-colour matches can split the edge of one white paper
        # into several components. Evaluate their union, never discard satellites.
        candidate = (
            _paper_quad_from_evidence(board, raw, cell, ignored_mask)
            if profile.board_mode == "visible_plane"
            else None
        )
        line_candidate = candidate is not None
        if candidate is None and profile.board_mode == "visible_plane":
            candidate = _paper_seed_from_projection(board, predicted, parity, model, ignored_mask)
            line_candidate = True
        elif candidate is None:
            candidate = _quad_from_hull(np.concatenate(contours))
        if candidate is not None:
            if profile.board_mode == "visible_plane":
                candidate = _refine_paper_boundaries(
                    candidate, board, predicted, parity, model, ignored_mask
                )
                if (
                    np.any(candidate < 0)
                    or np.any(candidate[:, 0] >= width)
                    or np.any(candidate[:, 1] >= height)
                ):
                    raise ArtworkExtractError("画作四角超出可见平面，拒绝将底板外圈送入生成")
            candidate_area = cv2.contourArea(candidate)
            fill = sum(cv2.contourArea(contour) for contour in contours) / max(candidate_area, 1)
            angles = []
            for i in range(4):
                a, b = candidate[i - 1] - candidate[i], candidate[(i + 1) % 4] - candidate[i]
                angles.append(abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))
            # A square/circular frame drawn over the board is not an opaque sheet:
            # its centre may expose real checker cells. Detect a genuine 2x2
            # alternating block inside the candidate before filling its support.
            candidate_mask = np.zeros_like(cleaned)
            cv2.fillConvexPoly(candidate_mask, np.round(candidate).astype(int), 255)
            gx_values = range(int(np.ceil(-ox / sx)), int(np.floor((width - ox) / sx)))
            gy_values = range(int(np.ceil(-oy / sy)), int(np.floor((height - oy) / sy)))
            matched_cells = np.zeros((len(gy_values), len(gx_values)), dtype=bool)
            for yi, gy in enumerate(gy_values):
                for xi, gx in enumerate(gx_values):
                    x0, x1 = int(ox + (gx + 0.25) * sx), int(ox + (gx + 0.75) * sx)
                    y0, y1 = int(oy + (gy + 0.25) * sy), int(oy + (gy + 0.75) * sy)
                    inside = candidate_mask[y0:y1, x0:x1]
                    patch = distance[y0:y1, x0:x1]
                    if inside.size and np.all(inside > 0) and np.mean(patch < tolerance) > 0.92:
                        matched_cells[yi, xi] = True
            visible_internal_checker = bool(
                np.any(
                    matched_cells[:-1, :-1]
                    & matched_cells[1:, :-1]
                    & matched_cells[:-1, 1:]
                    & matched_cells[1:, 1:]
                )
            )
            if (
                (fill >= profile.rectangle_fill_ratio or line_candidate)
                and max(angles) < 0.40
                and not visible_internal_checker
            ):
                artwork_quad = candidate
                cleaned = candidate_mask
                if profile.board_mode == "visible_plane":
                    # A single homography cannot model tiny optical curvature.
                    # Remove only checker-colour components connected to the
                    # outside of the support and confined to its uncertain edge.
                    # Interior black marks, white space and coloured marks stay.
                    expected_gray = cv2.cvtColor(predicted.astype(np.float32), cv2.COLOR_BGR2GRAY)
                    actual_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    dark = int(np.argmin([np.median(expected_gray[parity == i]) for i in (0, 1)]))
                    # Identify one *measured* dark square. Small local phase
                    # errors can put its inner fringe in the predicted white
                    # phase; its exterior bulk must still verify the dark phase.
                    dark_prediction = design @ model["coefficients"][dark]
                    dark_distance = np.max(
                        np.abs(board.astype(np.float32) - dark_prediction), axis=2
                    )
                    match = (dark_distance < tolerance * 1.20).astype(np.uint8)
                    opening_size = max(3, int(cell * 0.06) | 1)
                    opening_kernel = np.ones((opening_size, opening_size), np.uint8)
                    separated = cv2.morphologyEx(match, cv2.MORPH_OPEN, opening_kernel)
                    count_bg, bg_labels, bg_stats, _ = cv2.connectedComponentsWithStats(
                        separated, connectivity=4
                    )
                    depth = cv2.distanceTransform(candidate_mask, cv2.DIST_L2, 3)
                    max_depth = max(2.0, cell * 0.35)
                    dark_limit = float(np.median(expected_gray[parity == dark])) + 35
                    context = (
                        ((distance < tolerance * 1.2) | (match > 0))
                        & ((candidate_mask == 0) | (depth < max_depth + 4))
                    ).astype(np.uint8)
                    context = cv2.morphologyEx(context, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    _, context_labels = cv2.connectedComponents(context, connectivity=4)
                    exterior_labels = np.unique(
                        context_labels[(candidate_mask == 0) & (context > 0)]
                    )
                    exterior_labels = exterior_labels[exterior_labels > 0]
                    externally_connected = np.isin(context_labels, exterior_labels)
                    verified_background = np.zeros_like(candidate_mask)
                    for label in range(1, count_bg):
                        bx, by, bw, bh, area = bg_stats[label]
                        if area < 12:
                            continue
                        pad = opening_size // 2 + 1
                        x0, x1 = max(0, bx - pad), min(width, bx + bw + pad)
                        y0, y1 = max(0, by - pad), min(height, by + bh + pad)
                        region = np.s_[y0:y1, x0:x1]
                        component = (
                            cv2.dilate(
                                (bg_labels[region] == label).astype(np.uint8), opening_kernel
                            )
                            > 0
                        )
                        component &= match[region] > 0
                        support = candidate_mask[region] > 0
                        inside = component & support
                        if not np.any(inside) or not np.any(
                            component & externally_connected[region]
                        ):
                            continue
                        outside = component & ~support
                        phase_evidence = (
                            outside
                            if np.count_nonzero(outside)
                            >= max(20, np.count_nonzero(component) * 0.10)
                            else component
                        )
                        if np.mean(parity[region][phase_evidence] == dark) < 0.55:
                            continue
                        single_square_shape = bw <= sx * 1.35 and bh <= sy * 1.35
                        # A black stroke continuing over the expected white
                        # checker phase is artwork, not a residual background cell.
                        neighbour = (
                            cv2.dilate(component.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                        )
                        continued_ink = (
                            neighbour
                            & support
                            & (parity[region] != dark)
                            & (grid_distance[region] > 0.15)
                            & (actual_gray[region] < dark_limit)
                        )
                        if np.count_nonzero(continued_ink) > 4 and not single_square_shape:
                            continue
                        deepest = float(np.max(depth[region][inside]))
                        if deepest > max_depth:
                            if deepest < cell * 0.90:
                                partial_checker_residual_pixels += int(np.count_nonzero(inside))
                                partial_checker_residual_depth = max(
                                    partial_checker_residual_depth, deepest
                                )
                            continue
                        cleaned[region][inside] = 0
                        verified_background[region][component] = 255
                        edge_checker_removed_pixels += int(np.count_nonzero(inside))
                    cleaned, edge_matte_sides, edge_matte_max_inset = _matte_support_edges(
                        cleaned, verified_background, candidate, cell, profile.max_board_side
                    )
    return (
        cleaned,
        artwork_quad,
        {
            "foreground_fraction": fraction,
            "component_count": len(components),
            "foreground_tolerance": tolerance,
            "visible_internal_checker": visible_internal_checker,
            "edge_checker_removed_pixels": edge_checker_removed_pixels,
            "partial_checker_residual_pixels": partial_checker_residual_pixels,
            "partial_checker_residual_depth": partial_checker_residual_depth,
            "edge_matte_sides": edge_matte_sides,
            "edge_matte_max_inset": edge_matte_max_inset,
            "artwork_method": "opaque_quadrilateral"
            if artwork_quad is not None
            else "spatial_components",
        },
    )


def extract_artwork(
    image_bgr: np.ndarray,
    *,
    profile: ExtractionProfile | dict | None = None,
    board_quad: np.ndarray | list | None = None,
) -> ArtworkExtractionResult:
    """Locate board, remove its pattern, then rectify/crop the retained artwork.

    ``board_quad`` is optional calibration in this exact input's coordinates.
    It is still validated against actual checker pixels.  A camera screenshot
    and native photograph must never share coordinates without a mapping.
    """
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or min(image_bgr.shape[:2]) < 64
    ):
        raise ArtworkExtractError("输入必须是不小于 64×64 的 uint8 BGR 图像")
    settings = _profile(profile)
    try:
        quad, board, matrix, checker, method = _locate_board(image_bgr, settings, board_quad)
        source_ui = np.zeros(image_bgr.shape[:2], np.uint8)
        source_ignore = np.zeros_like(source_ui)
        source_height, source_width = source_ui.shape
        for region in settings.ignore_regions:
            x0, y0, x1, y1 = np.round(region).astype(int)
            cv2.rectangle(source_ignore, (x0, y0), (x1, y1), 255, -1)
        for region in settings.preview_occlusion_regions:
            x0, y0, x1, y1 = np.round(
                np.array(region) * [source_width, source_height, source_width, source_height]
            ).astype(int)
            cv2.rectangle(source_ui, (x0, y0), (x1, y1), 255, -1)
        ignored = cv2.warpPerspective(
            source_ignore | source_ui,
            matrix,
            (board.shape[1], board.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        alpha, artwork_quad, foreground_meta = _foreground(board, checker, settings, ignored)
        board_ui = cv2.warpPerspective(
            source_ui, matrix, (board.shape[1], board.shape[0]), flags=cv2.INTER_NEAREST
        )
        ui_overlap_pixels = int(np.count_nonzero((board_ui > 0) & (alpha > 0)))
        residual_meta = _checker_residual_gate(board, alpha, checker, settings)
        residual_meta["checker_residual_detected"] |= bool(
            foreground_meta["partial_checker_residual_pixels"]
        )
        # A preview occlusion is returned first so the controller can fall back
        # to one native photograph. Native/clean inputs never bypass this gate.
        if residual_meta["checker_residual_detected"] and not ui_overlap_pixels:
            raise ArtworkExtractError(
                "提取结果仍含连续棋盘背景，已阻止提交生成，请重新定位纸张", details=residual_meta
            )
        bgra = cv2.cvtColor(board, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        if artwork_quad is not None:
            artwork, artwork_matrix = _warp(
                bgra, artwork_quad, settings.artwork_expected_ratio, settings.max_board_side
            )
            if not foreground_meta["edge_checker_removed_pixels"]:
                artwork[:, :, 3] = 255  # complete opaque support, including blank corners
        else:
            ys, xs = np.nonzero(alpha)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            artwork = bgra[y0:y1, x0:x1].copy()
            artwork_matrix = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        old_height, old_width = artwork.shape[:2]
        if settings.rotation_degrees:
            rotations = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            transforms = {
                90: [[0, -1, old_height - 1], [1, 0, 0], [0, 0, 1]],
                180: [[-1, 0, old_width - 1], [0, -1, old_height - 1], [0, 0, 1]],
                270: [[0, 1, 0], [-1, 0, old_width - 1], [0, 0, 1]],
            }
            artwork = cv2.rotate(artwork, rotations[settings.rotation_degrees])
            artwork_matrix = (
                np.array(transforms[settings.rotation_degrees], np.float64) @ artwork_matrix
            )
        final_alpha = artwork[:, :, 3].copy()
        weight = final_alpha[:, :, None].astype(np.float32) / 255
        model_input = np.round(artwork[:, :, :3] * weight + 255 * (1 - weight)).astype(np.uint8)
        artwork[final_alpha == 0, :3] = 0
        overlay = image_bgr.copy()
        cv2.polylines(overlay, [np.round(quad).astype(np.int32)], True, (40, 220, 40), 3)
        if artwork_quad is not None:
            original_art = cv2.perspectiveTransform(artwork_quad[None], np.linalg.inv(matrix))[0]
            cv2.polylines(
                overlay, [np.round(original_art).astype(np.int32)], True, (0, 190, 255), 3
            )
        else:
            contours, _ = cv2.findContours(alpha, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            inverse = np.linalg.inv(matrix)
            for contour in contours:
                if len(contour) >= 3:
                    original_art = cv2.perspectiveTransform(contour.astype(np.float32), inverse)
                    cv2.polylines(
                        overlay, [np.round(original_art).astype(np.int32)], True, (0, 190, 255), 2
                    )
        meta = {
            "source_size": [image_bgr.shape[1], image_bgr.shape[0]],
            "board_size": [board.shape[1], board.shape[0]],
            "output_size": [artwork.shape[1], artwork.shape[0]],
            "board_method": method,
            "checker_confidence": checker["confidence"],
            "checker_contrast": checker["contrast"],
            "checker_ring_agreement": checker["ring_agreement"],
            "checker_grid": checker["grid"],
            "rotation_degrees": settings.rotation_degrees,
            "preview_occlusion": bool(ui_overlap_pixels),
            "preview_occlusion_pixels": ui_overlap_pixels,
            "preview_occlusion_regions": list(settings.preview_occlusion_regions)
            if ui_overlap_pixels
            else [],
            "source_to_board": matrix.tolist(),
            "board_to_artwork": artwork_matrix.tolist(),
            "artwork_corners_space": "rectified_board",
            "profile": asdict(settings),
            **foreground_meta,
            **residual_meta,
        }
        return ArtworkExtractionResult(
            board_image=board,
            artwork=artwork,
            rgba=cv2.cvtColor(artwork, cv2.COLOR_BGRA2RGBA),
            model_input=model_input,
            overlay=overlay,
            mask=final_alpha,
            board_mask=alpha,
            board_corners=quad.astype(float).tolist(),
            artwork_corners=None if artwork_quad is None else artwork_quad.astype(float).tolist(),
            meta=meta,
        )
    except ArtworkExtractError:
        raise
    except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
        raise ArtworkExtractError(f"画作提取失败: {exc}") from exc
