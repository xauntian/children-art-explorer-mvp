"""Fast fiducial extraction: geometry, real alpha, and fail-closed contracts."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.artwork_extract import ArtworkExtractError
from app.fast_artwork_extract import extract_fast_artwork

ROOT = Path(__file__).resolve().parents[2]


def board():
    image = cv2.imdecode(
        np.frombuffer((ROOT / "assets/capture-board-fast.png").read_bytes(), np.uint8), 1
    )
    assert image is not None
    return image


def paper():
    image = board()
    image[340:1230, 260:940] = 255
    cv2.rectangle(image, (275, 355), (925, 1215), (0, 0, 0), 9)
    cv2.circle(image, (600, 780), 120, (0, 0, 0), -1)
    cv2.circle(image, (600, 780), 70, (255, 255, 255), -1)
    cv2.line(image, (310, 440), (880, 1120), (20, 40, 235), 9)
    return image


def scene(image):
    h, w = image.shape[:2]
    source = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    quad = np.array([[110, 105], [1190, 65], [1260, 1645], [55, 1680]], np.float32)
    transform = cv2.getPerspectiveTransform(source, quad)
    photo = cv2.warpPerspective(image, transform, (1310, 1732), borderValue=(45, 60, 65))
    return photo, quad


def test_full_opaque_paper_preserves_black_white_and_white_hole_in_drawing():
    result = extract_fast_artwork(paper(), profile={"rotation_degrees": 0})
    assert result.meta["board_method"] == "aruco_known_board"
    assert result.meta["artwork_method"] == "opaque_quadrilateral"
    assert result.model_input.shape[1] / result.model_input.shape[0] == pytest.approx(
        680 / 890, abs=0.02
    )
    assert np.all(result.mask[10:-10, 10:-10] == 255)
    assert np.count_nonzero(np.all(result.model_input < 20, axis=2)) > 5000
    assert np.count_nonzero(np.all(result.model_input > 240, axis=2)) > 50000
    assert result.meta["checker_residual_detected"] is False
    assert np.count_nonzero(result.board_mask[:100]) == 0


def test_perspective_native_source_independently_finds_all_marker_geometry():
    photo, expected = scene(paper())
    result = extract_fast_artwork(photo)
    assert np.max(np.abs(np.asarray(result.board_corners) - expected)) < 3
    assert result.meta["marker_reprojection_rms_px"] < 1.5
    assert result.model_input.shape[1] / result.model_input.shape[0] == pytest.approx(
        890 / 680, abs=0.03
    )
    assert np.all(result.mask[12:-12, 12:-12] == 255)


def test_blur_jpeg_and_smooth_exposure_change_do_not_require_period_estimation():
    photo, _ = scene(paper())
    yy, xx = np.mgrid[: photo.shape[0], : photo.shape[1]]
    illumination = 0.75 + 0.18 * xx / photo.shape[1] + 0.12 * yy / photo.shape[0]
    photo = np.clip(photo.astype(float) * illumination[:, :, None] + 5, 0, 255).astype(np.uint8)
    photo = cv2.GaussianBlur(photo, (3, 3), 0.7)
    encoded = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
    photo = cv2.imdecode(encoded, 1)
    result = extract_fast_artwork(photo)
    assert result.meta["background_fit_error_p85"] < 16
    assert np.count_nonzero(np.all(result.model_input < 40, axis=2)) > 4000
    assert np.all(result.mask[15:-15, 15:-15] == 255)


def test_ring_center_remains_transparent_even_with_closed_outer_contour():
    image = board()
    cv2.circle(image, (600, 760), 260, (240, 240, 240), 85)
    result = extract_fast_artwork(image, profile={"rotation_degrees": 0})
    assert result.meta["artwork_method"] == "spatial_components"
    h, w = result.mask.shape
    assert result.mask[h // 2, w // 2] == 0
    assert np.count_nonzero(result.mask) > 30000
    assert np.all(result.model_input[h // 2, w // 2] == 255)


def test_rectangular_ring_cannot_be_mistaken_for_opaque_paper():
    image = board()
    cv2.rectangle(image, (300, 400), (900, 1200), (0, 0, 0), 50)
    result = extract_fast_artwork(image, profile={"rotation_degrees": 0})
    assert result.meta["artwork_method"] == "spatial_components"
    assert result.mask[result.mask.shape[0] // 2, result.mask.shape[1] // 2] == 0


def test_separate_artwork_elements_are_all_retained_without_largest_rectangle_rule():
    image = board()
    cv2.circle(image, (350, 520), 60, (0, 0, 240), -1)
    cv2.rectangle(image, (740, 1050), (850, 1150), (230, 80, 10), -1)
    result = extract_fast_artwork(image, profile={"rotation_degrees": 0})
    assert result.meta["component_count"] == 2
    assert result.model_input.shape[0] > 400 and result.model_input.shape[1] > 300
    assert np.count_nonzero(result.mask == 0) > 50000
    assert np.any(result.model_input[:, :, 2] > result.model_input[:, :, 0] + 80)


@pytest.mark.parametrize("marker", [0, 1, 2, 3])
def test_any_missing_marker_fails_closed(marker):
    image = paper()
    x, y = [(96, 96), (1104, 96), (1104, 1504), (96, 1504)][marker]
    image[y - 80 : y + 80, x - 80 : x + 80] = 120
    with pytest.raises(ArtworkExtractError, match="markers"):
        extract_fast_artwork(image)


def test_non_coplanar_or_misplaced_marker_fails_geometry_gate():
    image = paper()
    tile = image[24:168, 1032:1176].copy()
    image[24:168, 1032:1176] = 120
    image[80:224, 1000:1144] = tile
    with pytest.raises(ArtworkExtractError, match="plane|reprojection"):
        extract_fast_artwork(image)


def test_blank_board_is_not_successful_artwork():
    with pytest.raises(ArtworkExtractError, match="No distinct artwork"):
        extract_fast_artwork(board())


def test_wrong_background_pattern_is_not_accepted_just_because_ids_match():
    image = board()
    image[180:1420] = 120
    image[340:1230, 260:940] = 255
    with pytest.raises(ArtworkExtractError, match="background"):
        extract_fast_artwork(image)


def test_artwork_extending_outside_safe_zone_is_rejected_without_cropping():
    image = paper()
    image[400:1000, 140:300] = 255
    with pytest.raises(ArtworkExtractError, match="zone"):
        extract_fast_artwork(image)


def test_ui_intersection_is_metadata_for_controller_photo_fallback():
    image = paper()
    result = extract_fast_artwork(
        image, profile={"preview_occlusion_regions": [[0.3, 0.65, 0.7, 0.7]]}
    )
    assert result.meta["preview_occlusion"] is True
    clean = extract_fast_artwork(
        image, profile={"preview_occlusion_regions": [[0.3, 0.84, 0.7, 0.87]]}
    )
    assert clean.meta["preview_occlusion"] is False


def test_asset_demo_uses_exact_spec_and_returns_upright_whole_artwork():
    image = cv2.imdecode(
        np.frombuffer((ROOT / "assets/capture-board-fast-demo.png").read_bytes(), np.uint8), 1
    )
    result = extract_fast_artwork(
        image, profile={"board_spec_path": str(ROOT / "assets/capture-board-fast.json")}
    )
    assert result.model_input.shape[1] > result.model_input.shape[0]
    assert np.all(result.mask[10:-10, 10:-10] == 255)
    assert result.meta["marker_ids"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "image", [None, np.zeros((1, 1, 3), np.uint8), np.zeros((500, 500), np.uint8)]
)
def test_invalid_input_fails_closed(image):
    with pytest.raises(ArtworkExtractError):
        extract_fast_artwork(image)


def test_invalid_configuration_is_a_classified_error():
    with pytest.raises(ArtworkExtractError):
        extract_fast_artwork(board(), profile={"rotation_degrees": 45})


@pytest.mark.parametrize("size", [(1288, 1732), (768, 1024), (3072, 4096)])
def test_resampled_preview_and_native_sized_marker_frames(size):
    image = cv2.imdecode(
        np.frombuffer((ROOT / "assets/capture-board-fast-demo.png").read_bytes(), np.uint8), 1
    )
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    result = extract_fast_artwork(image)
    assert result.meta["artwork_method"] == "opaque_quadrilateral"
    assert result.model_input.shape[1] / result.model_input.shape[0] == pytest.approx(
        1120 / 844, abs=0.02
    )
    assert result.meta["marker_reprojection_rms_px"] < 1.5
    assert np.all(result.mask[10:-10, 10:-10] == 255)


def test_real_preview_has_complete_paper_and_no_false_corner_foreground():
    path = Path(__file__).resolve().parent / "fixtures/capture/fast-board-preview.png"
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1)
    result = extract_fast_artwork(image)
    assert result.meta["marker_reprojection_rms_px"] < 1.5
    assert result.model_input.shape[1] / result.model_input.shape[0] == pytest.approx(
        1080 / 814, abs=0.03
    )
    assert np.all(result.mask[8:-8, 8:-8] == 255)
    assert np.count_nonzero(result.board_mask[:150]) == 0
    assert np.count_nonzero(result.board_mask[:, :120]) == 0
    assert np.count_nonzero(result.board_mask[:, 650:]) == 0
    assert result.meta["preview_occlusion"] is False
