"""OpenCV document scanning for an opaque paper on a dark, plain background.

Uses standard Canny/Otsu contours, convex quadrilaterals, measured four-side
contrast, and colour-preserving warpPerspective. No foreground colour keying
or semantic image generation is performed. The approach follows the public
OpenCV document-scanning pattern also used by jscanify; no third-party source
code is copied.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .artwork_extract import ArtworkExtractError, ArtworkExtractionResult


@dataclass
class PaperExtractionProfile:
    max_detection_side: int = 1000
    max_output_side: int = 1024
    rotation_degrees: int = 270
    preview_occlusion_regions: tuple = ()


def _profile(value):
    try:
        if value is None:
            p = PaperExtractionProfile()
        elif isinstance(value, PaperExtractionProfile):
            p = PaperExtractionProfile(**asdict(value))
        elif isinstance(value, dict):
            p = PaperExtractionProfile(**value)
        else:
            raise ValueError("profile must be a configuration object")
        if not 400 <= p.max_detection_side <= 2000 or not 256 <= p.max_output_side <= 2048:
            raise ValueError("invalid detection/output size")
        if p.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("rotation_degrees must be 0, 90, 180 or 270")
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
        raise ArtworkExtractError(f"Invalid paper extraction profile: {exc}") from exc
    return p


def _order(points):
    q = np.asarray(points, np.float32)
    center = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0]))]
    return np.roll(q, -np.argmin(q.sum(1)), axis=0)


def _sample(image, points):
    return cv2.remap(
        image,
        points[:, 0].astype(np.float32)[None],
        points[:, 1].astype(np.float32)[None],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[0]


def _side_samples(q, index, count=100):
    a, b = q[index], q[(index + 1) % 4]
    direction = b - a
    normal = np.array([-direction[1], direction[0]], np.float32)
    normal /= np.linalg.norm(normal)
    if np.dot(q.mean(0) - (a + b) / 2, normal) < 0:
        normal = -normal
    points = a + (b - a) * np.linspace(0.025, 0.975, count)[:, None]
    return points.astype(np.float32), normal


def _evaluate(gray, ui, q):
    h, w = gray.shape
    area = cv2.contourArea(q)
    if area < h * w * 0.12 or not cv2.isContourConvex(q):
        return None
    if (
        np.any(q[:, 0] < 1.5)
        or np.any(q[:, 0] > w - 2.5)
        or np.any(q[:, 1] < 1.5)
        or np.any(q[:, 1] > h - 2.5)
    ):
        return None
    sides = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)
    if min(sides) < min(h, w) * 0.18 or max(sides) / min(sides) > 4:
        return None
    for i in range(4):
        u = q[(i - 1) % 4] - q[i]
        v = q[(i + 1) % 4] - q[i]
        angle = np.degrees(
            np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
        )
        if not 45 < angle < 135:
            return None
    inner = cv2.perspectiveTransform(
        np.array(
            [
                [
                    [0.15, 0.15],
                    [0.5, 0.15],
                    [0.85, 0.15],
                    [0.15, 0.5],
                    [0.5, 0.5],
                    [0.85, 0.5],
                    [0.15, 0.85],
                    [0.5, 0.85],
                    [0.85, 0.85],
                ]
            ],
            np.float32,
        ),
        cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), q),
    )[0]
    white = float(np.quantile(_sample(gray, inner), 0.75))
    if white < 100:
        return None
    evidence = []
    for i in range(4):
        points, normal = _side_samples(q, i)
        inside = points + normal * 3
        outside = points - normal * 3
        far = points + normal * min(24.0, max(9.0, sides[i] * 0.035))
        valid = (
            (outside[:, 0] >= 0)
            & (outside[:, 0] < w - 1)
            & (outside[:, 1] >= 0)
            & (outside[:, 1] < h - 1)
        )
        valid &= (
            (_sample(ui, points) == 0)
            & (_sample(ui, inside) == 0)
            & (_sample(ui, outside) == 0)
            & (_sample(ui, far) == 0)
        )
        if np.count_nonzero(valid) < 35:
            return None
        a = _sample(gray, inside)[valid].astype(float)
        b = _sample(gray, outside)[valid].astype(float)
        deep = _sample(gray, far)[valid].astype(float)
        delta = a - b
        continuity = float(np.mean(np.abs(delta) > 12))
        contrast = float(np.median(np.abs(delta)))
        outer_brightness = []
        for offset in (10.0, 16.0):
            farther_outside = points - normal * offset
            farther_valid = (
                (farther_outside[:, 0] >= 0)
                & (farther_outside[:, 0] < w - 1)
                & (farther_outside[:, 1] >= 0)
                & (farther_outside[:, 1] < h - 1)
                & (_sample(ui, farther_outside) == 0)
            )
            # Internal ink can look like a dark background immediately next to
            # its edge. A further exposed band must remain dark too. Near the
            # camera boundary only bands actually inside the image are used.
            outer_brightness.extend(_sample(gray, farther_outside)[valid & farther_valid].tolist())
        background = float(np.quantile(b, 0.7))
        if len(outer_brightness) > 20 and np.quantile(outer_brightness, 0.7) > white * 0.68:
            return None
        # A black pen border is allowed to meet the paper edge. Its reverse
        # contrast is evidence too; the interior paper itself remains opaque.
        paper_rim = float(np.mean(np.maximum(a, deep) > np.maximum(b + 20, white * 0.45)))
        if (
            continuity < 0.60
            or contrast < 20
            or background > white * 0.68
            or (paper_rim < 0.45 and np.median(delta) > -15)
        ):
            return None
        evidence.append(
            {
                "continuity": continuity,
                "contrast": contrast,
                "visible_fraction": float(np.mean(valid)),
                "outside_luma": background,
            }
        )
    score = float(np.mean([e["continuity"] for e in evidence]))
    return score, evidence


def _candidate_quads(gray, ui):
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(smooth, 50, 200)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bright = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    quads = []
    for mask in (bright, edges):
        mask[ui > 0] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(
            contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3], reverse=True
        )[:20]:
            if cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3] < gray.size * 0.12:
                continue
            hull = cv2.convexHull(contour)
            if cv2.contourArea(hull) < gray.size * 0.12:
                continue
            for epsilon in (0.004, 0.008, 0.015, 0.025):
                polygon = cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * epsilon, True)
                if len(polygon) != 4:
                    continue
                q = _order(polygon.reshape(4, 2))
                if cv2.contourArea(hull) < cv2.contourArea(q) * 0.97:
                    continue
                if not any(np.max(np.linalg.norm(q - prev, axis=1)) < 5 for prev in quads):
                    quads.append(q)
                break
    return quads


def _refine(gray, ui, q):
    lines = []
    for i in range(4):
        points, normal = _side_samples(q, i, 140)
        offsets = np.arange(-4.0, 4.1, 0.5)
        values = np.asarray([_sample(gray, points + normal * offset) for offset in offsets], float)
        gradients = np.abs(np.diff(values, axis=0))
        indices = np.argmax(gradients, axis=0)
        peaks = points + normal * (offsets[indices] + 0.25)[:, None]
        valid = (
            (_sample(ui, points) == 0)
            & (_sample(ui, peaks) == 0)
            & (gradients[indices, np.arange(len(points))] > 6)
        )
        if np.count_nonzero(valid) < 45:
            return q
        observed = peaks[valid]
        vx, vy, x, y = cv2.fitLine(observed, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        direction = np.array([vx, vy])
        origin = np.array([x, y])
        distances = np.abs((observed[:, 0] - x) * vy - (observed[:, 1] - y) * vx)
        if np.mean(distances < 1.5) < 0.80:
            return q
        lines.append((origin, direction))
    result = []
    for i in range(4):
        a, u = lines[(i - 1) % 4]
        b, v = lines[i]
        matrix = np.column_stack((u, -v))
        if abs(np.linalg.det(matrix)) < 0.1:
            return q
        result.append(a + np.linalg.solve(matrix, b - a)[0] * u)
    result = np.asarray(result, np.float32)
    return result if np.max(np.linalg.norm(result - q, axis=1)) < 7 else q


def extract_paper_artwork(image_bgr: np.ndarray, *, profile=None) -> ArtworkExtractionResult:
    """Find a complete four-sided paper, preserve all pixels, and rectify it."""
    start = time.perf_counter()
    p = _profile(profile)
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or min(image_bgr.shape[:2]) < 200
    ):
        raise ArtworkExtractError("Expected a uint8 BGR camera image")
    h, w = image_bgr.shape[:2]
    scale = min(1.0, p.max_detection_side / max(h, w))
    small = (
        cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image_bgr
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ui = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in p.preview_occlusion_regions:
        cv2.rectangle(
            ui,
            (int(x0 * gray.shape[1]), int(y0 * gray.shape[0])),
            (int(np.ceil(x1 * gray.shape[1])), int(np.ceil(y1 * gray.shape[0]))),
            255,
            -1,
        )
    candidates = []
    for q in _candidate_quads(gray, ui):
        evidence = _evaluate(gray, ui, q)
        if evidence is not None:
            candidates.append((q, *evidence))
    if not candidates:
        raise ArtworkExtractError(
            "No complete paper quadrilateral has reliable evidence on all four sides"
        )
    candidates.sort(key=lambda c: cv2.contourArea(c[0]), reverse=True)
    q, score, evidence = candidates[0]
    for other, _, _ in candidates[1:]:
        distance = np.mean(np.linalg.norm(other - q, axis=1))
        if distance > min(gray.shape) * 0.04 and cv2.contourArea(other) > cv2.contourArea(q) * 0.60:
            raise ArtworkExtractError(
                "More than one plausible paper boundary; capture is ambiguous"
            )
    refined = _refine(gray, ui, q)
    checked = _evaluate(gray, ui, refined)
    if checked is not None:
        q = refined
        score, evidence = checked
    source = q / scale
    outw = max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3]))
    outh = max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1]))
    output_scale = min(1.0, p.max_output_side / max(outw, outh))
    ow, oh = max(2, round(outw * output_scale)), max(2, round(outh * output_scale))
    target = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    # warpPerspective's linear interpolation is not an antialiasing filter
    # for a several-fold native-photo reduction. Average source pixels first,
    # retaining extra sampling headroom for perspective and final rotation.
    warp_scale = min(1.0, output_scale * 1.25)
    warp_source = (
        cv2.resize(image_bgr, None, fx=warp_scale, fy=warp_scale, interpolation=cv2.INTER_AREA)
        if warp_scale < 1
        else image_bgr
    )
    warp_matrix = transform @ np.diag([1 / warp_scale, 1 / warp_scale, 1.0])
    paper = cv2.warpPerspective(warp_source, warp_matrix, (ow, oh), flags=cv2.INTER_LINEAR)
    source_mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(source_mask, np.rint(q).astype(int), 255)
    preview_occlusion = bool(np.any((source_mask > 0) & (ui > 0)))
    artwork = np.dstack((paper, np.full((oh, ow), 255, np.uint8)))
    if p.rotation_degrees:
        artwork = cv2.rotate(
            artwork,
            {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[
                p.rotation_degrees
            ],
        )
    overlay = image_bgr.copy()
    cv2.polylines(
        overlay, [np.rint(source).astype(int)], True, (0, 180, 255), max(2, round(max(h, w) / 600))
    )
    model = artwork[:, :, :3].copy()
    alpha = artwork[:, :, 3]
    board_to_artwork = {
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, -1, oh - 1], [1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, ow - 1], [0, -1, oh - 1], [0, 0, 1]],
        270: [[0, 1, 0], [-1, 0, ow - 1], [0, 0, 1]],
    }[p.rotation_degrees]
    meta = {
        "board_method": "paper_document_scan",
        "artwork_method": "opaque_paper_quadrilateral",
        "rotation_degrees": p.rotation_degrees,
        "preview_occlusion": preview_occlusion,
        "confidence": score,
        "side_evidence": evidence,
        "candidate_count": len(candidates),
        "source_to_board": transform.tolist(),
        "board_to_artwork": board_to_artwork,
        "extraction_seconds": time.perf_counter() - start,
        "alpha_policy": "whole_opaque_paper",
    }
    return ArtworkExtractionResult(
        paper,
        artwork,
        cv2.cvtColor(artwork, cv2.COLOR_BGRA2RGBA),
        model,
        overlay,
        alpha,
        np.full((oh, ow), 255, np.uint8),
        source.tolist(),
        target.tolist(),
        meta,
    )
