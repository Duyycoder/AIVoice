# -*- coding: utf-8 -*-
"""Test cổng bỏ face detailer khi mặt quá nhỏ trong frame cuối."""
import os
import sys

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.models import CharacterLayer  # noqa: E402
from app.services.storytelling.studio.studio_pipeline import (  # noqa: E402
    _face_too_small_to_detail)


def _layer(scale, framing):
    return CharacterLayer(slug="a", prompt="p", scale=scale, framing=framing)


def test_wide_shot_skips_detailer():
    # wide: scale 0.45 trên khung cao 432 → lớp ~194px, mặt ~23px < 30px
    assert _face_too_small_to_detail(_layer(0.45, "full"), 432) is True


def test_close_shot_keeps_detailer():
    # close: scale 1.0 trên khung cao 432 → mặt ~181px, rất đáng vẽ lại
    assert _face_too_small_to_detail(_layer(1.0, "close"), 432) is False


def test_medium_shot_keeps_detailer():
    # medium: scale 0.8 → lớp ~346px, mặt ~69px
    assert _face_too_small_to_detail(_layer(0.8, "medium"), 432) is False


def test_taller_output_frame_can_rescue_a_wide_shot():
    # Khung cao hơn → cùng scale vẫn cho mặt đủ lớn, không được bỏ detailer
    assert _face_too_small_to_detail(_layer(0.45, "full"), 1080) is False


def test_bad_scale_value_does_not_skip():
    layer = CharacterLayer(slug="a", prompt="p", framing="full")
    layer.scale = "oops"
    assert _face_too_small_to_detail(layer, 432) is False
