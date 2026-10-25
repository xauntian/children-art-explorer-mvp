"""Real geometric extraction contracts, without cameras or generation services."""

from pathlib import Path

import cv2
import numpy as np
import pytest

import app.artwork_extract as extraction
from app.artwork_extract import ArtworkExtractError, ExtractionProfile, extract_artwork


def checker(size=512, cells=16):
    y, x = np.mgrid[:size, :size]
    gray = np.where(((x * cells // size) + (y * cells // size)) % 2, 240, 25).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def with_paper(size=512, cells=16):
    board = checker(size, cells)
    board[140:370, 130:390] = (255, 255, 255)
    cv2.rectangle(board, (165, 180), (350, 325), (0, 0, 0), 5)
    cv2.line(board, (175, 200), (320, 300), (10, 15, 220), 7)
    return board


def scene(board, quad=None):
    height, width = board.shape[:2]
    photo = np.full((650, 730, 3), (75, 90, 100), np.uint8)
    if quad is None:
        quad = np.array([[80, 65], [591, 65], [591, 576], [80, 576]], np.float32)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    matrix = cv2.getPerspectiveTransform(source, quad)
    warped = cv2.warpPerspective(board, matrix, (photo.shape[1], photo.shape[0]))
    valid = cv2.warpPerspective(
        np.full(board.shape[:2], 255, np.uint8), matrix, (photo.shape[1], photo.shape[0])
    )
    photo[valid > 0] = warped[valid > 0]
    return photo, quad


def profile(**overrides):
    return ExtractionProfile(checker_cells=(16, 16), **overrides)


def test_automatically_finds_board_and_preserves_whole_opaque_artwork():
    photo, expected = scene(with_paper())
    result = extract_artwork(photo, profile=profile())
    assert np.max(np.abs(np.asarray(result.board_corners) - expected)) < 8
    assert result.meta["board_method"] == "detected"
    assert result.meta["artwork_method"] == "opaque_quadrilateral"
    assert result.model_input.shape[2] == 3
    assert result.artwork.shape[2] == result.rgba.shape[2] == 4
    assert np.min(result.mask) == 255
    assert abs(result.artwork.shape[1] / result.artwork.shape[0] - 260 / 230) < 0.12
    assert np.any(np.all(result.model_input < 50, axis=2)), "black drawing lines must survive"
    signed = result.model_input.astype(np.int16)
    assert np.any(signed[:, :, 2] > signed[:, :, 0] + 80), "colour must survive"
    assert np.all(result.model_input[result.model_input.shape[0] // 2, -8] > 230), (
        "paper blank space must survive"
    )
    assert np.count_nonzero(result.board_mask[:50]) == 0


def test_board_perspective_and_rotated_paper_are_corrected_without_fixed_trim():
    paper = np.full((220, 260, 3), 255, np.uint8)
    cv2.rectangle(paper, (4, 4), (255, 215), (15, 70, 230), 5)
    cv2.line(paper, (25, 35), (220, 180), (0, 0, 0), 6)
    board = checker()
    paper_quad = cv2.boxPoints(((256, 256), (260, 220), 14)).astype(np.float32)
    source = np.array([[0, 0], [259, 0], [259, 219], [0, 219]], np.float32)
    # boxPoints order is irrelevant to the extractor, but make the test paper upright.
    paper_quad = paper_quad[np.argsort(np.arctan2(paper_quad[:, 1] - 256, paper_quad[:, 0] - 256))]
    matrix = cv2.getPerspectiveTransform(source, paper_quad)
    art = cv2.warpPerspective(paper, matrix, (512, 512))
    valid = cv2.warpPerspective(np.full(paper.shape[:2], 255, np.uint8), matrix, (512, 512))
    board[valid > 0] = art[valid > 0]
    photo, quad = scene(board, np.array([[65, 42], [630, 90], [580, 610], [105, 555]], np.float32))
    result = extract_artwork(
        photo, profile=profile(artwork_expected_ratio=260 / 220), board_quad=quad
    )
    assert result.meta["artwork_method"] == "opaque_quadrilateral"
    assert result.artwork.shape[1] / result.artwork.shape[0] == pytest.approx(260 / 220, abs=0.01)
    assert (
        np.count_nonzero((result.model_input[:, :, 2] > 150) & (result.model_input[:, :, 0] < 100))
        > 700
    )
    assert result.meta["board_to_artwork"] != np.eye(3).tolist()


def test_disconnected_elements_are_all_kept_and_background_is_transparent():
    board = checker()
    cv2.circle(board, (145, 230), 42, (5, 20, 220), -1)
    cv2.circle(board, (365, 270), 24, (220, 60, 10), -1)
    result = extract_artwork(board, profile=profile())
    assert result.meta["component_count"] == 2
    assert result.meta["artwork_method"] == "spatial_components"
    assert np.any(result.mask == 0) and np.any(result.mask == 255)
    opaque = result.mask > 0
    assert np.any((result.model_input[:, :, 2] > 180) & (result.model_input[:, :, 0] < 60) & opaque)
    assert np.any((result.model_input[:, :, 0] > 180) & (result.model_input[:, :, 2] < 60) & opaque)
    assert np.all(result.model_input[result.mask == 0] == 255)
    assert np.all(result.artwork[result.mask == 0, :3] == 0)


def test_empty_checker_is_rejected_instead_of_returning_fake_artwork():
    with pytest.raises(ArtworkExtractError, match="未检测到"):
        extract_artwork(checker(), profile=profile())


def test_nonchecker_photo_fails_even_with_explicit_quad():
    image = np.full((512, 512, 3), 180, np.uint8)
    quad = [[0, 0], [511, 0], [511, 511], [0, 511]]
    with pytest.raises(ArtworkExtractError, match="棋盘"):
        extract_artwork(image, board_quad=quad, profile=profile())


def test_wrong_source_calibration_and_invalid_input_fail_closed():
    with pytest.raises(ArtworkExtractError, match="超出"):
        extract_artwork(
            with_paper(), board_quad=[[0, 0], [600, 0], [600, 600], [0, 600]], profile=profile()
        )
    with pytest.raises(ArtworkExtractError, match="BGR"):
        extract_artwork(np.zeros((512, 512, 4), np.uint8))
    with pytest.raises(ArtworkExtractError, match="重复"):
        extract_artwork(with_paper(), board_quad=[[1, 1]] * 4, profile=profile())


def test_default_profile_matches_32_cell_template_and_does_not_mutate_input():
    image = with_paper(cells=32)
    original = image.copy()
    result = extract_artwork(image)
    assert np.array_equal(image, original)
    assert result.meta["profile"]["checker_cells"] == (32, 32)
    assert result.meta["checker_confidence"] > 0.75


def test_invalid_profile_is_actionable():
    with pytest.raises(ArtworkExtractError, match="配置"):
        extract_artwork(with_paper(), profile={"board_expected_ratio": 0})


def test_opaque_support_keeps_separate_black_white_marks_and_internal_blank():
    board = checker()
    board[150:360, 130:380] = 255
    cv2.circle(board, (180, 240), 18, (0, 0, 0), -1)
    cv2.rectangle(board, (290, 200), (335, 270), (0, 0, 0), 4)
    result = extract_artwork(board, profile=profile())
    assert result.meta["artwork_method"] == "opaque_quadrilateral"
    assert np.all(result.mask == 255)
    assert np.count_nonzero(np.all(result.model_input < 25, axis=2)) > 900
    assert np.count_nonzero(np.all(result.model_input > 245, axis=2)) > 30000


def test_automatic_detection_handles_perspective_blur_and_uneven_light():
    board = with_paper(cells=32)
    # Photographs of screens do not reproduce the SVG's colours exactly.
    y, x = np.mgrid[:512, :512]
    illumination = (0.72 + 0.20 * x / 512 - 0.08 * y / 512)[:, :, None]
    board = np.clip(board * illumination + 9, 0, 255).astype(np.uint8)
    board = cv2.GaussianBlur(board, (3, 3), 0.65)
    expected = np.array([[65, 42], [630, 90], [580, 610], [105, 555]], np.float32)
    photo, _ = scene(board, expected)
    ok, encoded = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    photo = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    result = extract_artwork(photo)
    assert np.max(np.abs(np.asarray(result.board_corners) - expected)) < 10
    assert result.meta["checker_ring_agreement"] > 0.85
    assert result.meta["artwork_method"] == "opaque_quadrilateral"
    assert np.all(result.mask == 255)


def test_wrong_checker_profile_is_rejected_instead_of_guessing_foreground():
    with pytest.raises(ArtworkExtractError, match="棋盘|网格"):
        extract_artwork(with_paper(cells=16), profile=ExtractionProfile(checker_cells=(30, 30)))


@pytest.mark.parametrize("shape", ["circle", "square"])
def test_a_closed_artwork_loop_does_not_make_visible_checker_in_its_hole_opaque(shape):
    board = checker()
    if shape == "circle":
        cv2.circle(board, (256, 256), 125, (15, 40, 220), 34)
    else:
        cv2.rectangle(board, (125, 125), (385, 385), (15, 40, 220), 32)
    result = extract_artwork(board, profile=profile())
    assert result.meta["artwork_method"] == "spatial_components"
    # Transform a board-space point in the genuinely exposed centre to output.
    matrix = np.asarray(result.meta["board_to_artwork"], np.float32)
    centre = cv2.perspectiveTransform(np.array([[[256, 256]]], np.float32), matrix)[0, 0]
    x, y = np.round(centre).astype(int)
    assert np.all(result.mask[y - 25 : y + 25, x - 25 : x + 25] == 0)
    assert np.all(result.model_input[y - 25 : y + 25, x - 25 : x + 25] == 255)
    assert np.count_nonzero(result.mask) > 15000


def cropped_plane():
    height, width, pitch = 780, 600, 40
    yy, xx = np.mgrid[:height, :width]
    parity = (np.floor((xx - 13) / pitch).astype(int) + np.floor((yy - 22) / pitch).astype(int)) % 2
    gray = np.where(parity, 235, 20).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image[90:690, 55:550] = 250
    cv2.circle(image, (95, 140), 18, (15, 30, 220), -1)
    cv2.circle(image, (510, 640), 18, (220, 50, 15), -1)
    cv2.rectangle(image, (180, 270), (400, 500), (0, 0, 0), 5)
    return image


def visible_profile(**extra):
    return {
        "board_mode": "visible_plane",
        "checker_cells": None,
        "board_expected_ratio": None,
        **extra,
    }


def test_partial_plane_estimates_unknown_pitch_phase_and_preserves_paper_edges():
    result = extract_artwork(cropped_plane(), profile=visible_profile(rotation_degrees=270))
    assert result.meta["board_method"] == "visible_plane_grid"
    assert result.meta["checker_grid"]["visible_intersections"] >= 18
    assert result.meta["rotation_degrees"] == 270
    assert result.model_input.shape[1] / result.model_input.shape[0] == pytest.approx(
        600 / 495, abs=0.06
    )
    assert result.meta["preview_occlusion"] is False
    assert np.all(result.mask[10:-10, 10:-10] == 255), "the opaque paper interior must stay opaque"
    assert (
        np.count_nonzero((result.model_input[:, :, 2] > 180) & (result.model_input[:, :, 0] < 60))
        > 600
    )
    assert (
        np.count_nonzero((result.model_input[:, :, 0] > 180) & (result.model_input[:, :, 2] < 60))
        > 600
    )
    # Checker cells may be far outside the crop. They must not pad the paper.
    assert np.mean(result.model_input[8:15, 40:-40]) > 230
    assert np.mean(result.model_input[-15:-8, 40:-40]) > 230


def test_camera_overlay_overlap_is_reported_without_painting_over_occluded_art():
    image = cropped_plane()
    box = [270 / 600, 660 / 780, 330 / 600, 720 / 780]
    cv2.circle(image, (300, 690), 27, (60, 60, 60), -1)
    cv2.putText(image, "2x", (283, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    result = extract_artwork(image, profile=visible_profile(preview_occlusion_regions=[box]))
    assert result.meta["preview_occlusion"] is True
    assert result.meta["preview_occlusion_pixels"] > 100
    assert result.meta["preview_occlusion_regions"]
    clean_photo = extract_artwork(cropped_plane(), profile=visible_profile())
    assert clean_photo.meta["preview_occlusion"] is False


def test_visible_plane_still_rejects_nonchecker_and_invalid_normalized_ui_boxes():
    with pytest.raises(ArtworkExtractError, match="棋盘|周期"):
        extract_artwork(np.full((780, 600, 3), 190, np.uint8), profile=visible_profile())
    with pytest.raises(ArtworkExtractError, match="UI"):
        extract_artwork(
            cropped_plane(), profile=visible_profile(preview_occlusion_regions=[[0, 0, 1.2, 1]])
        )


def test_visible_plane_edge_cleanup_keeps_black_border_strokes_and_coloured_marks():
    image = cropped_plane()
    cv2.rectangle(image, (56, 92), (548, 688), (0, 0, 0), 5)
    result = extract_artwork(image, profile=visible_profile())
    gray = cv2.cvtColor(result.model_input, cv2.COLOR_BGR2GRAY)
    assert np.count_nonzero(gray[:10] < 70) > 800
    assert np.count_nonzero(gray[-10:] < 70) > 800
    assert (
        np.count_nonzero((result.model_input[:, :, 2] > 180) & (result.model_input[:, :, 0] < 60))
        > 600
    )
    assert np.all(result.mask[20:-20, 20:-20] == 255)


def test_missing_hough_side_uses_measured_projection_and_never_a_hull(monkeypatch):
    monkeypatch.setattr(extraction, "_paper_quad_from_evidence", lambda *args: None)
    monkeypatch.setattr(
        extraction,
        "_quad_from_hull",
        lambda *args: pytest.fail("visible-plane paper must not fall back to a hull"),
    )
    result = extract_artwork(cropped_plane(), profile=visible_profile())
    inverse = np.linalg.inv(np.asarray(result.meta["source_to_board"]))
    corners = cv2.perspectiveTransform(
        np.asarray(result.artwork_corners, np.float32)[None], inverse
    )[0]
    expected = np.array([[55, 90], [549, 90], [549, 689], [55, 689]])
    assert np.max(np.abs(corners - expected)) < 10
    assert result.meta["checker_residual_detected"] is False


def test_residual_gate_detects_a_single_cell_wide_checker_rim():
    image = checker()
    settings = profile()
    model = extraction._checker_model(image, settings)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[32:64, 32:480] = 255  # one row, so an old 2x2-only guard cannot detect it
    verdict = extraction._checker_residual_gate(image, mask, model, settings)
    assert verdict["checker_residual_detected"] is True
    assert verdict["checker_residual_max_run"] >= 3


def test_bad_retained_background_is_blocked_before_model_input_but_ui_can_fall_back(monkeypatch):
    original = extraction._foreground

    def bad_support(board, model, settings, ignored):
        _, _, meta = original(board, model, settings, ignored)
        h, w = board.shape[:2]
        return (
            np.full((h, w), 255, np.uint8),
            np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32),
            meta,
        )

    monkeypatch.setattr(extraction, "_foreground", bad_support)
    with pytest.raises(ArtworkExtractError, match="连续棋盘"):
        extract_artwork(cropped_plane(), profile=visible_profile())
    # The controller needs this metadata to take one clean native photograph.
    preview = extract_artwork(
        cropped_plane(),
        profile=visible_profile(preview_occlusion_regions=[[0.45, 0.85, 0.55, 0.93]]),
    )
    assert preview.meta["preview_occlusion"] is True
    assert preview.meta["checker_residual_detected"] is True


@pytest.mark.parametrize(
    "filename",
    [
        "checkerboard-native-2x.jpg",
        "checkerboard-20261024T182256-c53c4316.jpg",
        "checkerboard-20261024T192206-92a7bf91.jpg",
    ],
)
def test_real_huawei_captures_have_no_checker_rim_and_keep_opaque_interior(filename):
    path = Path(__file__).resolve().parent / "fixtures" / "capture" / filename
    photo = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    result = extract_artwork(photo, profile=visible_profile(rotation_degrees=270))
    assert result.meta["checker_residual_detected"] is False
    assert result.meta["partial_checker_residual_pixels"] == 0
    assert result.meta["preview_occlusion"] is False
    assert np.all(result.mask[60:-60, 60:-60] == 255)
    assert 1.25 < result.model_input.shape[1] / result.model_input.shape[0] < 1.40
    if "c53c4316" in filename:
        # Annotated exterior of this real paper: the old bad extraction retained
        # dark AND white checker fragments here. Both phases must be transparent.
        assert np.mean(result.mask[10:300, -8:] < 80) > 0.90
        assert np.all(result.mask[30:300, -55:-45] == 255)
