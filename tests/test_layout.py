"""Offline checks for the Figma print template and its JPEG encoding."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFont

from mijia1s.printer import MAX_JPEG_BYTES, encode_mijia_image
from mijia1s.layout import (
    ARTWORK_BOX,
    ASSET_DIR,
    DESCRIPTION_FONT_SIZE,
    DESCRIPTION_POSITION,
    NAME_POSITION,
    PHOTO_SIZE,
    TEXT_WIDTH,
    TITLE_FONT_SIZE,
    caption_font,
    compose_print_image,
    print_caption,
)

STYLES = json.loads((ASSET_DIR / "styles.json").read_text(encoding="utf-8"))
COLOURS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))


def marker_artwork(path: Path, size: tuple[int, int]) -> None:
    """Distinct corner marks expose cropping, mirroring, and wrong rotation."""
    width, height = size
    image = Image.new("RGB", size, (120, 120, 120))
    draw = ImageDraw.Draw(image)
    for box, colour in zip(
        ((0, 0, width // 4, height // 4),
         (width * 3 // 4, 0, width - 1, height // 4),
         (0, height * 3 // 4, width // 4, height - 1),
         (width * 3 // 4, height * 3 // 4, width - 1, height - 1)),
        COLOURS,
        strict=True,
    ):
        draw.rectangle(box, fill=colour)
    image.save(path)


def ink_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    return ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()


def test_template_assets_and_bundled_variable_font_are_available():
    assert (ASSET_DIR / "figma-background.png").is_file()
    assert (ASSET_DIR / "Inter.ttf").is_file()
    assert (ASSET_DIR / "OFL.txt").is_file()
    with Image.open(ASSET_DIR / "figma-background.png") as background:
        assert background.format == "PNG"
        assert background.width > 0 and background.height > 0
    font = ImageFont.truetype(str(ASSET_DIR / "Inter.ttf"), 32)
    axes = font.get_variation_axes()
    assert any(axis["name"] == b"Weight" and axis["minimum"] <= 400
               and axis["maximum"] >= 900 for axis in axes)


def test_all_eight_styles_have_short_print_copy():
    assert len(STYLES) == 8
    assert len({style["id"] for style in STYLES}) == 8


@pytest.mark.parametrize("style", STYLES, ids=lambda style: style["id"])
def test_style_caption_stays_on_one_line_inside_photo(style):
    name, description = print_caption(style)
    assert name == style["print_name"]
    assert description == style["print_description"]
    assert all(character.isascii() for character in name + description)
    for text, position, bold in ((name, NAME_POSITION, True),
                                 (description, DESCRIPTION_POSITION, False)):
        assert "\n" not in text and "\r" not in text
        font = caption_font(text, bold=bold)
        assert (48 <= font.size <= TITLE_FONT_SIZE) if bold else (24 <= font.size <= DESCRIPTION_FONT_SIZE)
        assert font.getlength(text) <= TEXT_WIDTH
        left, top, right, bottom = font.getbbox(text, anchor="lt")
        assert position[0] + left >= 0
        assert position[0] + right <= PHOTO_SIZE[0]
        assert right - left <= TEXT_WIDTH
        assert position[1] + top >= ARTWORK_BOX[3]
        assert position[1] + bottom <= PHOTO_SIZE[1]
        assert bottom - top <= (70 if bold else 40)


def test_exact_ai_artwork_is_placed_at_original_pixels_with_correct_orientation(tmp_path):
    source = tmp_path / "output.png"
    size = (1303, 1006)
    expected_corners = (COLOURS[2], COLOURS[0], COLOURS[3], COLOURS[1])
    marker_artwork(source, size)
    original = source.read_bytes()
    page = compose_print_image(source, STYLES[0])
    assert page.size == (1040, 1560)
    assert page.mode == "RGB"
    artwork = page.crop(ARTWORK_BOX)
    bounds = ink_box(artwork)
    assert bounds is not None
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    assert (width, height) == (1006, 1303)
    assert height == ARTWORK_BOX[3] - ARTWORK_BOX[1]
    for (x_fraction, y_fraction), colour in zip(
        ((0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)),
        expected_corners,
        strict=True,
    ):
        point = (left + round((width - 1) * x_fraction),
                 top + round((height - 1) * y_fraction))
        assert artwork.getpixel(point) == colour
    assert source.read_bytes() == original
    with Image.open(source) as unchanged:
        assert unchanged.size == size
        assert ImageChops.difference(
            artwork, unchanged.transpose(Image.Transpose.ROTATE_270)
        ).getbbox() is None


def test_legacy_image_uses_proportional_cover_without_stretching(tmp_path):
    source = tmp_path / "legacy.png"
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).ellipse((170, 70, 230, 130), fill="blue")
    image.save(source)
    page = compose_print_image(source, STYLES[0])
    mask = page.crop(ARTWORK_BOX).getchannel("R").point(lambda p: 255 if p < 128 else 0)
    left, top, right, bottom = mask.getbbox()
    assert abs((right - left) - (bottom - top)) <= 2
    assert 300 <= right - left <= 314


@pytest.mark.parametrize("name", ["Impressionism", "Post-Impressionism", "Expressionism", "Pop Art"])
def test_large_title_preserves_descenders(tmp_path, name):
    source = tmp_path / "white.png"
    Image.new("RGB", (1303, 1006), "white").save(source)
    page = compose_print_image(source, {"print_name": name, "print_description": "Short description."})
    actual = page.crop((0, 1338, 1040, 1438)).convert("L")
    reference = Image.new("L", (1040, 100), 255)
    ImageDraw.Draw(reference).text((2, 4), name, font=caption_font(name, bold=True),
                                  fill=0, anchor="lt")
    actual_ink = sum(actual.histogram()[:250])
    reference_ink = sum(reference.histogram()[:250])
    assert actual_ink >= reference_ink * 0.999


def test_transparent_source_background_is_composited_on_white(tmp_path):
    source = tmp_path / "transparent.png"
    image = Image.new("RGBA", (200, 400), (255, 0, 255, 0))
    ImageDraw.Draw(image).rectangle((75, 150, 125, 250), fill=(0, 0, 255, 255))
    image.save(source)
    original = source.read_bytes()
    page = compose_print_image(source, STYLES[0])
    left, top, right, bottom = ARTWORK_BOX
    assert page.mode == "RGB"
    assert page.getpixel(((left + right) // 2, top + 50)) == (255, 255, 255)
    assert page.getpixel(((left + right) // 2, (top + bottom) // 2)) == (0, 0, 255)
    assert source.read_bytes() == original


@pytest.mark.parametrize("style", [
    {},
    {"print_name": "Fauvism"},
    {"print_name": " ", "print_description": "Valid description."},
    {"print_name": 42, "print_description": "Valid description."},
    {"print_name": "Fauvism\nCubism", "print_description": "Valid description."},
    {"print_name": "Fauvism", "print_description": "Two\nlines"},
    {"print_name": "Fauvism", "print_description": "Two\rlines"},
    {"print_name": "Fauvism", "print_description": " "},
    {"print_name": "Fauvism", "print_description": ["not text"]},
    {"print_name": "Fauvism", "print_description": "W" * 2000},
])
def test_invalid_caption_fails_before_creating_layout_file(tmp_path, style):
    source = tmp_path / "output.png"
    marker_artwork(source, (200, 400))
    original = source.read_bytes()
    with pytest.raises(ValueError):
        encode_mijia_image(source, style_snapshot=style)
    assert not (tmp_path / "output-print.png").exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize("style", [None, [], "Fauvism"])
def test_missing_or_nonmapping_caption_is_rejected(style):
    with pytest.raises(ValueError):
        print_caption(style)


def test_encoder_saves_text_template_and_bounded_jpeg_without_changing_source(tmp_path):
    source = tmp_path / "output.png"
    marker_artwork(source, (400, 200))
    original = source.read_bytes()
    snapshot = dict(next(style for style in STYLES if style["id"] == "post_impressionism"))
    encoded = encode_mijia_image(source, style_snapshot=snapshot)
    assert 0 < len(encoded) <= MAX_JPEG_BYTES == 500 * 1024
    with Image.open(io.BytesIO(encoded)) as jpeg:
        assert jpeg.format == "JPEG"
        assert jpeg.size == (1040, 1560)
        assert jpeg.mode == "RGB"
    output = tmp_path / "output-print.png"
    assert output.is_file()
    with Image.open(output) as template:
        assert template.format == "PNG"
        assert template.size == (1040, 1560)
        assert template.mode == "RGB"
        assert ink_box(template.crop((37, 1360, 1003, 1435))) is not None
        assert ink_box(template.crop((37, 1442, 1003, 1490))) is not None
        assert ink_box(template.crop(ARTWORK_BOX)) is not None
    assert source.read_bytes() == original
