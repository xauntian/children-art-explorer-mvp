"""Known four-marker board extraction without periodic-grid estimation.

The marker board defines geometry; measured exposed background defines colour.
Only the physical artwork support becomes opaque. Paper interiors are preserved,
while holes between independent or ring-shaped pieces remain transparent.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .artwork_extract import ArtworkExtractError, ArtworkExtractionResult


@dataclass
class FastExtractionProfile:
    board_spec_path: str | None = None
    max_detection_side: int = 1400
    max_board_side: int = 1024
    rotation_degrees: int = 270
    foreground_tolerance: float = 25.0
    min_artwork_pixels: int = 100
    preview_occlusion_regions: tuple = ()


def _profile(value):
    try:
        if value is None:
            p = FastExtractionProfile()
        elif isinstance(value, FastExtractionProfile):
            p = FastExtractionProfile(**asdict(value))
        elif isinstance(value, dict):
            p = FastExtractionProfile(**value)
        else:
            raise ValueError("profile must be a configuration object")
        if p.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("rotation_degrees must be 0, 90, 180 or 270")
        if not 384 <= p.max_board_side <= 2048 or not 640 <= p.max_detection_side <= 2400:
            raise ValueError("invalid detection or board size")
        if not 8 <= p.foreground_tolerance <= 50 or p.min_artwork_pixels < 10:
            raise ValueError("invalid foreground thresholds")
        p.preview_occlusion_regions = tuple(
            tuple(float(v) for v in b) for b in p.preview_occlusion_regions
        )
        if any(
            len(b) != 4
            or not np.all(np.isfinite(b))
            or min(b) < 0
            or max(b) > 1
            or b[2] <= b[0]
            or b[3] <= b[1]
            for b in p.preview_occlusion_regions
        ):
            raise ValueError("invalid normalized preview occlusion box")
    except (TypeError, ValueError) as exc:
        raise ArtworkExtractError(f"Invalid fast extraction profile: {exc}") from exc
    return p


def _board_spec(path):
    spec = {
        "board_size_px": [1200, 1600],
        "aruco": {
            "dictionary": "DICT_4X4_50",
            "markers": [
                {
                    "id": i,
                    "corners_px": [
                        [x - 56, y - 56],
                        [x + 56, y - 56],
                        [x + 56, y + 56],
                        [x - 56, y + 56],
                    ],
                    "quiet_bounds_ltrb": [x - 72, y - 72, x + 72, y + 72],
                }
                for i, (x, y) in enumerate(((96, 96), (1104, 96), (1104, 1504), (96, 1504)))
            ],
        },
        "background": {
            "checker_cell_px": 24,
            "origin_px": [0, 0],
            "colors_rgb": [[105, 119, 128], [124, 137, 145]],
        },
        "artwork_zone_ltrb": [176, 220, 1024, 1340],
    }
    if path:
        try:
            spec = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ArtworkExtractError(f"Cannot read marker board specification: {exc}") from exc
    try:
        w, h = (int(v) for v in spec["board_size_px"])
        markers = spec["aruco"]["markers"]
        if spec["aruco"]["dictionary"] != "DICT_4X4_50" or len(markers) != 4:
            raise ValueError("four DICT_4X4_50 markers required")
        if sorted(int(m["id"]) for m in markers) != [0, 1, 2, 3]:
            raise ValueError("marker IDs must be 0, 1, 2, 3")
        if min(w, h) < 400 or max(w, h) > 5000:
            raise ValueError("invalid board dimensions")
        for m in markers:
            corners = np.asarray(m["corners_px"], np.float32)
            if (
                corners.shape != (4, 2)
                or not np.all(np.isfinite(corners))
                or cv2.contourArea(corners) < 100
            ):
                raise ValueError("invalid marker corners")
        zone = np.asarray(spec["artwork_zone_ltrb"], np.float32)
        if zone.shape != (4,) or not (0 < zone[0] < zone[2] < w and 0 < zone[1] < zone[3] < h):
            raise ValueError("invalid artwork zone")
        if float(spec["background"]["checker_cell_px"]) < 8:
            raise ValueError("invalid background cell size")
        if np.asarray(spec["background"]["colors_rgb"]).shape != (2, 3):
            raise ValueError("two RGB background colours required")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtworkExtractError(f"Invalid marker board specification: {exc}") from exc
    return spec


def _geometry(image, spec, profile):
    h, w = image.shape[:2]
    scale = min(1.0, profile.max_detection_side / max(h, w))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 4
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
    )
    # Otsu suppresses the deliberately low-contrast grid before marker decoding.
    # Native exposure can put the white quiet zone below Otsu's split, so the
    # grayscale adaptive detector remains an independent fallback.
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    found = {}
    for detection_image in (binary, gray):
        detected, ids, _ = detector.detectMarkers(detection_image)
        if ids is None:
            continue
        seen = set()
        for corners, marker_id in zip(detected, ids.ravel(), strict=True):
            marker_id = int(marker_id)
            if marker_id in seen:
                raise ArtworkExtractError("Duplicate board marker ID")
            seen.add(marker_id)
            if marker_id not in found:
                points = corners.reshape(4, 2).astype(np.float32)
                cv2.cornerSubPix(
                    gray,
                    points,
                    (4, 4),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01),
                )
                found[marker_id] = points / scale
        if all(i in found for i in range(4)):
            break
    if not all(i in found for i in range(4)):
        raise ArtworkExtractError("All four board markers (0–3) must be visible")
    markers = sorted(spec["aruco"]["markers"], key=lambda m: m["id"])
    source = np.concatenate([found[m["id"]] for m in markers])
    target = np.concatenate([np.asarray(m["corners_px"], np.float32) for m in markers])
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.5 / scale)
    if matrix is None or inliers is None or np.count_nonzero(inliers) < 15:
        raise ArtworkExtractError("Board markers do not agree on one plane")
    predicted = cv2.perspectiveTransform(target[None], np.linalg.inv(matrix))[0]
    errors = np.linalg.norm(predicted - source, axis=1) * scale
    if np.max(errors) > 3.0 or np.sqrt(np.mean(errors**2)) > 1.5:
        raise ArtworkExtractError("Marker reprojection error is too large")
    bw, bh = spec["board_size_px"]
    board_corners = cv2.perspectiveTransform(
        np.asarray([[[0, 0], [bw, 0], [bw, bh], [0, bh]]], np.float32), np.linalg.inv(matrix)
    )[0]
    if not cv2.isContourConvex(board_corners) or cv2.contourArea(board_corners) < 0.08 * w * h:
        raise ArtworkExtractError("Marker board is too small or has invalid geometry")
    if (
        np.any(board_corners[:, 0] < -w * 0.02)
        or np.any(board_corners[:, 0] > w * 1.02)
        or np.any(board_corners[:, 1] < -h * 0.02)
        or np.any(board_corners[:, 1] > h * 1.02)
    ):
        raise ArtworkExtractError("The complete marker board must remain inside the preview")
    output_scale = profile.max_board_side / max(bw, bh)
    output_size = (round(bw * output_scale), round(bh * output_scale))
    transform = np.diag([output_scale, output_scale, 1.0]) @ matrix
    board = cv2.warpPerspective(image, transform, output_size, flags=cv2.INTER_LINEAR)
    return board, transform, board_corners, float(np.sqrt(np.mean(errors**2))), output_scale


def _background(board, spec, scale, ui_mask):
    h, w = board.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    origin = spec["background"].get("origin_px", [0, 0])
    cell = float(spec["background"]["checker_cell_px"]) * scale
    gx = (xx - origin[0] * scale) / cell
    gy = (yy - origin[1] * scale) / cell
    parity = (np.floor(gx).astype(int) + np.floor(gy).astype(int)) % 2
    distance = np.minimum.reduce((gx % 1, 1 - gx % 1, gy % 1, 1 - gy % 1))
    # Display viewing angle and lens shading are smooth but not necessarily
    # linear. Fit only exposed reference pixels, never the artwork interior.
    xn, yn = xx / w, yy / h
    design = np.stack((np.ones_like(xx), xn, yn, xn * xn, xn * yn, yn * yn), axis=-1)
    zone = np.asarray(spec["artwork_zone_ltrb"]) * scale
    zx0, zy0, zx1, zy1 = np.rint(zone).astype(int)
    calibration = np.ones((h, w), bool)
    calibration[max(0, zy0 - 5) : min(h, zy1 + 5), max(0, zx0 - 5) : min(w, zx1 + 5)] = False
    for marker in spec["aruco"]["markers"]:
        bounds = marker.get("quiet_bounds_ltrb")
        if bounds is None:
            pts = np.asarray(marker["corners_px"])
            bounds = [*pts.min(0) - 16, *pts.max(0) + 16]
        x0, y0, x1, y1 = np.rint(np.asarray(bounds) * scale).astype(int)
        calibration[max(0, y0 - 3) : min(h, y1 + 3), max(0, x0 - 3) : min(w, x1 + 3)] = False
    calibration &= (distance > 0.20) & (ui_mask == 0)
    prediction = np.zeros_like(board, dtype=np.float32)
    residuals = []
    colours = []
    for phase in (0, 1):
        ys, xs = np.where(calibration & (parity == phase))
        if len(ys) < 300:
            raise ArtworkExtractError("Not enough exposed reference background around the artwork")
        step = max(1, len(ys) // 5000)
        ys, xs = ys[::step], xs[::step]
        a = design[ys, xs]
        b = board[ys, xs].astype(np.float32)
        keep = np.ones(len(ys), bool)
        for _ in range(3):
            coefficients = np.linalg.lstsq(a[keep], b[keep], rcond=None)[0]
            residual = np.max(np.abs(b - a @ coefficients), axis=1)
            keep = residual < max(5.0, float(np.median(residual)) * 3)
            if np.count_nonzero(keep) < 200:
                raise ArtworkExtractError("Reference background is obscured or inconsistent")
        residuals.extend(residual.tolist())
        prediction[parity == phase] = (design @ coefficients)[parity == phase]
        colours.append(np.asarray([1.0, 0.5, 0.5, 0.25, 0.25, 0.25]) @ coefficients)
    error = float(np.quantile(residuals, 0.85))
    delta = colours[1] - colours[0]
    expected = np.diff(np.asarray(spec["background"]["colors_rgb"], np.float32), axis=0)[0, ::-1]
    contrast = float(np.linalg.norm(delta))
    alignment = float(np.dot(delta, expected) / max(np.linalg.norm(expected) * contrast, 1e-6))
    if error > 16 or contrast < 5 or alignment < 0.70:
        raise ArtworkExtractError(
            "The visible background does not match the calibrated marker board"
        )
    # Independent spatial checks prevent a flat replacement background from
    # borrowing its weak phase contrast from the remaining top/bottom checker.
    rails = (
        (xx < zx0 - 5) & (yy > zy0) & (yy < zy1),
        (xx > zx1 + 5) & (yy > zy0) & (yy < zy1),
        (yy < zy0 - 5) & (xx > zx0) & (xx < zx1),
        (yy > zy1 + 5) & (xx > zx0) & (xx < zx1),
    )
    for rail in rails:
        medians = []
        for phase in (0, 1):
            pixels = board[calibration & rail & (parity == phase)]
            if len(pixels) < 100:
                raise ArtworkExtractError("Not enough exposed background on every board side")
            medians.append(np.median(pixels, axis=0))
        local_delta = medians[1] - medians[0]
        local_contrast = float(np.linalg.norm(local_delta))
        local_alignment = float(
            np.dot(local_delta, expected) / max(np.linalg.norm(expected) * local_contrast, 1e-6)
        )
        if local_contrast < max(3.0, contrast * 0.30) or local_alignment < 0.70:
            raise ArtworkExtractError(
                "A board side does not contain the expected background pattern"
            )
    smooth = cv2.GaussianBlur(board.astype(np.float32), (3, 3), 0)
    reference = cv2.GaussianBlur(prediction, (3, 3), 0)
    difference = np.max(np.abs(smooth - reference), axis=2)
    return difference, (zx0, zy0, zx1, zy1), error, contrast


def _components(mask, min_pixels=8):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)
    kept = 0
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            result[labels == i] = 255
            kept += 1
    return result, kept


def _order(points):
    points = np.asarray(points, np.float32)
    center = points.mean(0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    return np.roll(points, -np.argmin(points.sum(1)), axis=0)


def _support(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    opaque_quad = None
    if len(contours) == 1:
        contour = contours[0]
        area = cv2.contourArea(contour)
        polygon = cv2.approxPolyDP(contour, max(1.0, cv2.arcLength(contour, True) * 0.006), True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon) and area > 100:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [contour], -1, 255, -1)
            missing = (filled > 0) & (mask == 0)
            # A real checker hole is never made opaque just because its outer
            # contour resembles a rectangle. Tiny isolated JPEG holes can fill.
            n, _, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), 8)
            largest_hole = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
            if np.count_nonzero(missing) < area * 0.015 and largest_hole < max(6, area * 0.0003):
                mask = filled
                opaque_quad = _order(polygon.reshape(4, 2))
    return mask, opaque_quad


def _rotate(image, degrees):
    if degrees == 0:
        return image
    return cv2.rotate(
        image,
        {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[
            degrees
        ],
    )


def extract_fast_artwork(image_bgr: np.ndarray, *, profile=None) -> ArtworkExtractionResult:
    """Extract from the exact four-marker board; ambiguous captures raise.

    The supplied image is a BGR camera-preview crop or a native photograph.
    No image/board transform from another source is accepted or reused.
    """
    start = time.perf_counter()
    p = _profile(profile)
    spec = _board_spec(p.board_spec_path)
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or min(image_bgr.shape[:2]) < 200
    ):
        raise ArtworkExtractError("Expected a nonempty uint8 BGR camera image")
    board, transform, corners, reprojection, scale = _geometry(image_bgr, spec, p)
    h, w = board.shape[:2]
    ui = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in p.preview_occlusion_regions:
        polygon = np.array(
            [
                [
                    [x0 * image_bgr.shape[1], y0 * image_bgr.shape[0]],
                    [x1 * image_bgr.shape[1], y0 * image_bgr.shape[0]],
                    [x1 * image_bgr.shape[1], y1 * image_bgr.shape[0]],
                    [x0 * image_bgr.shape[1], y1 * image_bgr.shape[0]],
                ]
            ],
            np.float32,
        )
        cv2.fillConvexPoly(
            ui, np.rint(cv2.perspectiveTransform(polygon, transform)[0]).astype(int), 255
        )
    difference, zone, bg_error, bg_contrast = _background(board, spec, scale, ui)
    mask = (difference > p.foreground_tolerance).astype(np.uint8) * 255
    x0, y0, x1, y1 = zone
    # Allow raster/subpixel interpolation at the printed placement guide.
    x0, y0, x1, y1 = x0 - 3, y0 - 3, x1 + 3, y1 + 3
    outside = mask.copy()
    outside[:2] = outside[-2:] = 0
    outside[:, :2] = outside[:, -2:] = 0
    outside[y0:y1, x0:x1] = 0
    outside[ui > 0] = 0
    for marker in spec["aruco"]["markers"]:
        pts = np.asarray(marker["corners_px"]) * scale
        bx0, by0 = np.floor(pts.min(0) - 18 * scale - 3).astype(int)
        bx1, by1 = np.ceil(pts.max(0) + 18 * scale + 3).astype(int)
        outside[max(0, by0) : min(h, by1), max(0, bx0) : min(w, bx1)] = 0
    outside, _ = _components(outside, 30)
    if np.count_nonzero(outside) > 100:
        raise ArtworkExtractError("Keep the entire artwork inside the marked artwork zone")
    mask[:y0] = 0
    mask[y1:] = 0
    mask[:, :x0] = 0
    mask[:, x1:] = 0
    mask, components = _components(mask)
    if np.count_nonzero(mask) < p.min_artwork_pixels:
        raise ArtworkExtractError("No distinct artwork is visible inside the board")
    if (
        np.any(mask[y0 : y0 + 2, x0:x1])
        or np.any(mask[y1 - 2 : y1, x0:x1])
        or np.any(mask[y0:y1, x0 : x0 + 2])
        or np.any(mask[y0:y1, x1 - 2 : x1])
    ):
        raise ArtworkExtractError("The artwork reaches the safe zone boundary; move it inward")
    mask, quad = _support(mask)
    preview_occlusion = bool(np.any((mask > 0) & (ui > 0)))
    if quad is not None:
        outw = round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))) + 1
        outh = round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))) + 1
        art_transform = cv2.getPerspectiveTransform(
            quad,
            np.asarray([[0, 0], [outw - 1, 0], [outw - 1, outh - 1], [0, outh - 1]], np.float32),
        )
        rgb = cv2.warpPerspective(board, art_transform, (outw, outh))
        alpha = cv2.warpPerspective(mask, art_transform, (outw, outh))
        artwork_corners = quad.tolist()
    else:
        ys, xs = np.where(mask > 0)
        ax0, ay0, ax1, ay1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        rgb = board[ay0:ay1, ax0:ax1].copy()
        alpha = mask[ay0:ay1, ax0:ax1].copy()
        art_transform = np.array([[1.0, 0.0, -ax0], [0.0, 1.0, -ay0], [0.0, 0.0, 1.0]])
        artwork_corners = None
    artwork = _rotate(np.dstack((rgb, alpha)), p.rotation_degrees)
    alpha = artwork[:, :, 3]
    a = alpha.astype(np.float32)[:, :, None] / 255
    model = np.rint(artwork[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    overlay = image_bgr.copy()
    cv2.polylines(overlay, [np.rint(corners).astype(int)], True, (0, 220, 0), 3)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inverse = np.linalg.inv(transform)
    for c in contour:
        original = cv2.perspectiveTransform(c.astype(np.float32), inverse)
        cv2.polylines(overlay, [np.rint(original).astype(int)], True, (0, 180, 255), 2)
    meta = {
        "board_method": "aruco_known_board",
        "board_spec_path": p.board_spec_path,
        "marker_ids": [0, 1, 2, 3],
        "marker_reprojection_rms_px": reprojection,
        "artwork_method": "opaque_quadrilateral" if quad is not None else "spatial_components",
        "component_count": components,
        "rotation_degrees": p.rotation_degrees,
        "preview_occlusion": preview_occlusion,
        "checker_residual_detected": False,
        "background_fit_error_p85": bg_error,
        "background_contrast": bg_contrast,
        "source_to_board": transform.tolist(),
        "board_to_artwork": art_transform.tolist(),
        "extraction_seconds": time.perf_counter() - start,
    }
    return ArtworkExtractionResult(
        board,
        artwork,
        cv2.cvtColor(artwork, cv2.COLOR_BGRA2RGBA),
        model,
        overlay,
        alpha,
        mask,
        corners.tolist(),
        artwork_corners,
        meta,
    )
