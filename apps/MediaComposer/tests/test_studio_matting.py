# -*- coding: utf-8 -*-
"""Test chroma-key matting — thuần numpy/PIL, không cần GPU."""
import os
import sys

import numpy as np
import pytest
from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.studio.matting import (  # noqa: E402
    ChromaMatter, GrabCutMatter, alpha_coverage, hex_to_rgb)

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
    """Despill chỉ giảm xanh ở pixel viền, không phá màu foreground đục."""
    m = ChromaMatter(bg_color=GREEN, despill=True, feather_px=0)
    edge = m.cutout(_img((20, 190, 70)))
    assert np.asarray(edge.convert("RGB"))[..., 1].max() < 190

    foreground = m.cutout(_img((200, 255, 200)))
    assert np.asarray(foreground.convert("RGB"))[..., 1].max() == 255


def test_despill_handles_all_dominant_magenta_channels():
    """Magenta có hai kênh trội; viền phải khử cả đỏ lẫn lam."""
    matter = ChromaMatter(bg_color="#FF00FF", despill=True, feather_px=0)
    edge = np.asarray(matter.cutout(_img((190, 20, 190))).convert("RGB"))

    assert edge[..., 0].max() < 190
    assert edge[..., 2].max() < 190


def test_grabcut_fallback_extracts_subject_from_non_chroma_background():
    pytest.importorskip("cv2")
    arr = np.full((120, 80, 3), (90, 100, 110), dtype=np.uint8)
    arr[20:115, 25:55] = (220, 30, 30)

    rgba = GrabCutMatter(iterations=3, feather_px=0).cutout(Image.fromarray(arr))
    cov = alpha_coverage(rgba)

    assert 0.15 < cov < 0.5
    assert rgba.getpixel((40, 60))[3] > 240
    assert rgba.getpixel((2, 2))[3] == 0
