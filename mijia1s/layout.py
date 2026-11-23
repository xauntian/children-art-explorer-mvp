"""Figma 15:2 的照片模板，按米家 1S 相纸像素排版。"""

from __future__ import annotations

from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PHOTO_SIZE = (1040, 1560)
LANDSCAPE_SIZE = (1560, 1040)
LANDSCAPE_ARTWORK_BOX = (32, 17, 1335, 1023)
ARTWORK_BOX = (17, 32, 1023, 1335)
NAME_POSITION = (37, 1366)
DESCRIPTION_POSITION = (37, 1442)
TITLE_FONT_SIZE = 64
DESCRIPTION_FONT_SIZE = 32
TEXT_WIDTH = 966
ASSET_DIR = Path(__file__).resolve().parent / "assets"


def print_caption(style_snapshot: dict) -> tuple[str, str]:
    """仅使用生成该作品时的画派快照，不把生成 Prompt 印成说明。"""
    if not isinstance(style_snapshot, dict):
        raise ValueError("缺少本轮画派资料，无法制作打印模板")
    name = style_snapshot.get("print_name") or style_snapshot.get("name")
    description = style_snapshot.get("print_description")
    for value in (name, description):
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n"):
            raise ValueError("打印模板需要有效的画派英文名和单行解释")
    return name.strip(), description.strip()


def caption_font(text: str, *, bold=False) -> ImageFont.FreeTypeFont:
    maximum = TITLE_FONT_SIZE if bold else DESCRIPTION_FONT_SIZE
    minimum = 48 if bold else 24
    for size in range(maximum, minimum - 1, -1):
        font = ImageFont.truetype(str(ASSET_DIR / "Inter.ttf"), size)
        font.set_variation_by_axes([14, 900 if bold else 400])
        if font.getlength(text) <= TEXT_WIDTH:
            return font
    raise ValueError("画派文字过长，请缩短为一行后再打印")


def _draw_rotated_caption(page: Image.Image, text: str, x: int, *, bold=False) -> None:
    font = caption_font(text, bold=bold)
    thickness = 58 if bold else 38
    width = ceil(font.getlength(text)) + 2
    bounds = font.getbbox(text, anchor="lt")
    ink_height = bounds[3] - bounds[1]
    strip_thickness = max(thickness, ink_height + 2)
    strip = Image.new("RGBA", (width, strip_thickness))
    top = (strip_thickness - ink_height) // 2
    ImageDraw.Draw(strip).text((1, top), text, font=font, fill="black", anchor="lt")
    column = strip.transpose(Image.Transpose.ROTATE_90)
    page.paste(column, (x + (thickness - strip_thickness) // 2,
                        1003 - column.height), column)


def compose_print_image(path: Path, style_snapshot: dict) -> Image.Image:
    """固定横向模板；1303×1006 AI 成品按原像素放入，再整页转为打印方向。"""
    name, description = print_caption(style_snapshot)
    with Image.open(ASSET_DIR / "figma-background.png") as background:
        page = background.convert("RGB").resize(LANDSCAPE_SIZE, Image.Resampling.LANCZOS)
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        image = Image.new("RGB", oriented.size, "white")
        image.paste(oriented, mask=oriented.getchannel("A"))
    left, top, right, bottom = LANDSCAPE_ARTWORK_BOX
    slot_size = (right - left, bottom - top)
    fitted = (image if image.size == slot_size else
              ImageOps.fit(image, slot_size, Image.Resampling.LANCZOS))
    page.paste(fitted, (left, top))
    _draw_rotated_caption(page, name, 1366, bold=True)
    _draw_rotated_caption(page, description, 1442)
    return page.transpose(Image.Transpose.ROTATE_270)
