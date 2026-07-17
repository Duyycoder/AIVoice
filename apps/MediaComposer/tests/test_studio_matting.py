# -*- coding: utf-8 -*-
"""Test chroma-key matting — thuần numpy/PIL, không cần GPU."""
import os
import sys

import numpy as np
from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.studio.matting import (  # noqa: E402
    ChromaMatter, alpha_coverage, hex_to_rgb)

GREEN = "#00B140"


def test_hex_to_rgb():
    assert hex_to_rgb("#00B140") == (0, 177, 64)
    assert hex_to_rgb("00b140") == (0, 177, 64)
    assert hex_to_rgb("bad") == (0, 177, 64)  # fallback


def _img(fill, size=(64, 64)):
    return Image.new("RGB", size, fill)


def test_solid_bg_is_transparent():
    """Ảnh toàn màu nền → alpha ~0 khắp nơi."""
    m = ChromaMatter(bg_color=GREEN)
    rgba = m.cutout(_img((0, 177, 64)))
    assert rgba.mode == "RGBA"
    assert alpha_coverage(rgba) < 0.02


def test_foreground_is_opaque():
    """Ảnh toàn màu đỏ (khác xa nền) → alpha ~1 khắp nơi."""
    m = ChromaMatter(bg_color=GREEN)
    rgba = m.cutout(_img((255, 0, 0)))
    assert alpha_coverage(rgba) > 0.98


def test_split_fg_bg():
    """Nửa đỏ / nửa xanh nền → coverage ~ 0.5."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :32] = (255, 0, 0)     # đỏ (giữ)
    arr[:, 32:] = (0, 177, 64)    # nền (bỏ)
    m = ChromaMatter(bg_color=GREEN, feather_px=0)
    cov = alpha_coverage(m.cutout(Image.fromarray(arr)))
    assert 0.4 < cov < 0.6


def test_despill_limits_bg_channel():
    """Despill giới hạn kênh trội của nền (xanh lá → kênh G)."""
    m = ChromaMatter(bg_color=GREEN, despill=True, feather_px=0)
    rgba = m.cutout(_img((200, 255, 200)))  # ám xanh
    g = np.asarray(rgba.convert("RGB"))[..., 1]
    assert g.max() <= 200  # G bị kẹp ≤ max(R,B)=200
