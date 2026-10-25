"""Opaque paper scanning preserves content and rejects uncertain boundaries."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.artwork_extract import ArtworkExtractError
from app.paper_extract import extract_paper_artwork

ROOT = Path(__file__).resolve().parents[2]
BOXES = (
    (0.26786, 0.86894, 0.35326, 0.93822),
    (0.38820, 0.86490, 0.48137, 0.94573),
    (0.51708, 0.86894, 0.60559, 0.93822),
    (0.64130, 0.86894, 0.73292, 0.93822),
    (0.86025, 0.86028, 0.95109, 0.94400),
)


def document():
    image = np.full((600, 450, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (429, 579), (0, 0, 0), 6)
    cv2.circle(image, (225, 300), 80, (0, 0, 0), -1)
    cv2.circle(image, (225, 300), 40, (255, 255, 255), -1)
    cv2.line(image, (30, 90), (420, 520), (20, 40, 235), 7)
    return image


def scene(paper=None, quad=None):
    if paper is None:
        paper = document()
    if quad is None:
        quad = np.array([[70, 60], [520, 35], [550, 665], [85, 680]], np.float32)
    image = np.full((720, 600, 3), 61, np.uint8)
    ph, pw = paper.shape[:2]
    matrix = cv2.getPerspectiveTransform(
        np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32), quad
    )
    warped = cv2.warpPerspective(paper, matrix, (600, 720))
    mask = cv2.warpPerspective(np.full(paper.shape[:2], 255, np.uint8), matrix, (600, 720))
    image[mask > 0] = warped[mask > 0]
    return image, quad


def test_perspective_paper_keeps_all_black_white_blank_and_colour():
    image, expected = scene()
    result = extract_paper_artwork(image, profile={"rotation_degrees": 0})
    assert np.max(np.abs(np.asarray(result.board_corners) - expected)) < 3
    assert np.all(result.mask == 255)
    assert np.any(np.all(result.model_input < 15, axis=2))
    assert np.any(np.all(result.model_input > 245, axis=2))
    assert np.any(
        result.model_input[:, :, 2].astype(int) > result.model_input[:, :, 0].astype(int) + 100
    )
    assert result.meta["preview_occlusion"] is False
    assert len(result.meta["side_evidence"]) == 4


def test_nearly_full_frame_with_few_pixels_of_background_has_no_zone():
    q = np.array([[7, 8], [590, 6], [592, 710], [9, 712]], np.float32)
    image, _ = scene(quad=q)
    result = extract_paper_artwork(image)
    assert np.max(np.abs(np.asarray(result.board_corners) - q)) < 3
    assert result.model_input.shape[1] > result.model_input.shape[0]
    assert np.all(result.mask == 255)


def test_dark_pen_marks_may_touch_every_paper_edge():
    paper = document()
    cv2.rectangle(paper, (0, 0), (449, 599), (0, 0, 0), 9)
    image, q = scene(paper)
    result = extract_paper_artwork(image, profile={"rotation_degrees": 0})
    assert np.max(np.abs(np.asarray(result.board_corners) - q)) < 4
    assert np.count_nonzero(np.all(result.model_input[:6] < 40, axis=2)) > 100
    assert np.all(result.mask == 255)


def test_internal_drawing_frame_is_not_selected_as_paper():
    paper = np.full((600, 450, 3), 255, np.uint8)
    cv2.rectangle(paper, (60, 70), (390, 540), (0, 0, 0), 10)
    image, q = scene(paper)
    result = extract_paper_artwork(image)
    assert np.max(np.abs(np.asarray(result.board_corners) - q)) < 3


def test_dark_monitor_border_is_not_selected_instead_of_inner_paper():
    image, q = scene()
    cv2.rectangle(image, (12, 12), (585, 705), (100, 100, 100), 4)
    result = extract_paper_artwork(image)
    assert np.max(np.abs(np.asarray(result.board_corners) - q)) < 3


def test_ui_over_paper_is_flagged_and_original_pixels_are_retained():
    image, q = scene()
    cv2.circle(image, (310, 670), 28, (100, 100, 100), -1)
    box = (0.46, 0.88, 0.59, 0.98)
    result = extract_paper_artwork(image, profile={"preview_occlusion_regions": [box]})
    assert result.meta["preview_occlusion"] is True
    assert np.max(np.abs(np.asarray(result.board_corners) - q)) < 4
    assert np.all(result.mask == 255)


def test_ui_outside_paper_does_not_trigger_photo_fallback():
    image, _ = scene()
    result = extract_paper_artwork(image, profile={"preview_occlusion_regions": [[0, 0.9, 0.1, 1]]})
    assert result.meta["preview_occlusion"] is False


def test_two_comparable_papers_are_ambiguous():
    image = np.full((700, 1000, 3), 61, np.uint8)
    image[80:620, 70:450] = 255
    image[70:610, 550:930] = 255
    with pytest.raises(ArtworkExtractError, match="ambiguous"):
        extract_paper_artwork(image)


def test_paper_cut_off_by_camera_edge_fails_closed():
    image = np.full((700, 600, 3), 61, np.uint8)
    image[0:650, 40:560] = 255
    with pytest.raises(ArtworkExtractError):
        extract_paper_artwork(image)


@pytest.mark.parametrize("value", [0, 61, 255])
def test_uniform_image_never_falls_back_to_full_image(value):
    with pytest.raises(ArtworkExtractError):
        extract_paper_artwork(np.full((700, 600, 3), value, np.uint8))


def test_low_contrast_uncertain_boundary_fails_closed():
    image = np.full((700, 600, 3), 130, np.uint8)
    image[60:640, 40:560] = 140
    with pytest.raises(ArtworkExtractError):
        extract_paper_artwork(image)


def test_missing_side_under_ui_cannot_be_invented():
    image, _ = scene()
    with pytest.raises(ArtworkExtractError):
        extract_paper_artwork(image, profile={"preview_occlusion_regions": [[0, 0.8, 1, 1]]})


@pytest.mark.parametrize("source", ["dark-board-live-a", "dark-board-native-a"])
def test_real_phone_paper_geometry_and_ui_contract(source):
    path = Path(__file__).resolve().parent / "fixtures/capture" / f"{source}.png"
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1)
    result = extract_paper_artwork(
        image, profile={"preview_occlusion_regions": BOXES if "live" in source else []}
    )
    assert result.meta["preview_occlusion"] is ("live" in source)
    assert result.model_input.shape[1] / result.model_input.shape[0] == pytest.approx(
        1080 / 814, abs=0.025
    )
    assert max(result.model_input.shape[:2]) <= 1024
    assert np.all(result.mask == 255)
    for side in result.meta["side_evidence"]:
        assert side["continuity"] > 0.9


def test_native_reduction_preserves_mean_colour_and_original_coordinate_transform():
    image, q = scene()
    large = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    result = extract_paper_artwork(large, profile={"max_output_side": 512, "rotation_degrees": 0})
    assert max(result.model_input.shape[:2]) == 512
    corners = np.asarray(result.board_corners, np.float32)
    assert np.max(np.abs(corners - q * 4)) < 12
    transformed = cv2.perspectiveTransform(
        corners[None], np.asarray(result.meta["source_to_board"])
    )[0]
    assert np.max(np.abs(transformed - np.asarray(result.artwork_corners))) < 0.1
    assert np.median(result.model_input.reshape(-1, 3), axis=0).min() > 240


@pytest.mark.parametrize(
    "image", [None, np.zeros((20, 20, 3), np.uint8), np.zeros((500, 500), np.uint8)]
)
def test_invalid_input_is_classified(image):
    with pytest.raises(ArtworkExtractError):
        extract_paper_artwork(image)


def test_invalid_profile_is_classified():
    with pytest.raises(ArtworkExtractError):
        extract_paper_artwork(scene()[0], profile={"rotation_degrees": 45})


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_source_board_artwork_composition_matches_actual_rotation(rotation):
    image, _ = scene()
    result = extract_paper_artwork(image, profile={"rotation_degrees": rotation})
    oh, ow = result.board_image.shape[:2]
    expected = {
        0: [[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]],
        90: [[oh - 1, 0], [oh - 1, ow - 1], [0, ow - 1], [0, 0]],
        180: [[ow - 1, oh - 1], [0, oh - 1], [0, 0], [ow - 1, 0]],
        270: [[0, ow - 1], [0, 0], [oh - 1, 0], [oh - 1, ow - 1]],
    }[rotation]
    composed = np.asarray(result.meta["board_to_artwork"]) @ np.asarray(
        result.meta["source_to_board"]
    )
    mapped = cv2.perspectiveTransform(np.asarray([result.board_corners], np.float32), composed)[0]
    assert np.max(np.abs(mapped - np.asarray(expected))) < 0.01
    board = result.board_image
    expected_pixels = (
        board
        if rotation == 0
        else cv2.rotate(
            board,
            {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[rotation],
        )
    )
    assert np.array_equal(result.model_input, expected_pixels)
    assert (
        np.max(
            np.abs(
                np.asarray(result.artwork_corners)
                - np.asarray([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])
            )
        )
        < 0.01
    )
