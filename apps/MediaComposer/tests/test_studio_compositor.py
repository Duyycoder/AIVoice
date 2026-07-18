# -*- coding: utf-8 -*-
"""Test toán đặt lớp + ghép ảnh — thuần PIL, không cần GPU."""
import os
import sys

from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.models import CharacterLayer  # noqa: E402
from app.services.storytelling.studio.compositor import (  # noqa: E402
    anchor_x_pos, anchor_y_pos, fit_layer_size, composite, trim_transparent)


def test_anchor_x():
    assert anchor_x_pos("left", 200, 50) == 0
    assert anchor_x_pos("center", 200, 50) == 75
    assert anchor_x_pos("right", 200, 50) == 150


def test_anchor_y():
    assert anchor_y_pos("bottom", 150, 100) == 50
    assert anchor_y_pos("middle", 150, 100) == 25


def test_fit_layer_size_keeps_ratio_and_clamps():
    # scale 0.5 của khung cao 400 → 200; giữ tỉ lệ 512x768
    tw, th = fit_layer_size(400, (512, 768), 0.5)
    assert th == 200
    assert tw == round(512 * 200 / 768)
    # scale > 1 bị kẹp về chiều cao khung
    _, th2 = fit_layer_size(400, (512, 768), 1.5)
    assert th2 == 400


def _rgba(fill, size):
    return Image.new("RGBA", size, fill)


def test_composite_places_layer_bottom_center():
    bg = Image.new("RGB", (200, 150), (0, 0, 255))          # nền xanh dương
    layer = _rgba((255, 0, 0, 255), (40, 60))               # lớp đỏ đục
    cl = CharacterLayer(slug="a", prompt="p", anchor_x="center",
                        anchor_y="bottom", scale=0.4, z_order=0)  # cao 60px
    out = composite(bg, [(layer, cl)], harmonize=False)
    assert out.size == (200, 150)
    # đáy giữa khung phải là đỏ
    r, g, b = out.getpixel((100, 148))
    assert r > 150 and g < 80 and b < 80
    # đỉnh khung vẫn là nền xanh dương
    assert out.getpixel((100, 2))[2] > 150


def test_composite_z_order():
    bg = Image.new("RGB", (100, 100), (255, 255, 255))
    red = _rgba((255, 0, 0, 255), (100, 100))
    green = _rgba((0, 255, 0, 255), (100, 100))
    back = CharacterLayer(slug="r", prompt="", scale=1.0, z_order=0)
    front = CharacterLayer(slug="g", prompt="", scale=1.0, z_order=5)
    out = composite(bg, [(red, back), (green, front)], harmonize=False)
    # lớp z lớn (xanh lá) nằm trên
    _, g, _ = out.getpixel((50, 50))
    assert g > 200


def test_trim_transparent_makes_scale_apply_to_visible_character():
    padded = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    padded.paste(Image.new("RGBA", (20, 40), (255, 0, 0, 255)), (40, 60))
    assert trim_transparent(padded).size == (20, 40)

    bg = Image.new("RGB", (100, 100), (0, 0, 255))
    layer = CharacterLayer(slug="a", prompt="", scale=0.5,
                           anchor_x="center", anchor_y="bottom")
    out = composite(bg, [(padded, layer)], harmonize=False)
    # Visible cutout cao đúng 50% frame; nếu scale canvas chroma thì điểm này còn nền.
    assert out.getpixel((50, 55))[0] > 200
