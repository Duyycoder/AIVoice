# -*- coding: utf-8 -*-
"""Smoke test StudioPipeline.render_plan — dùng ảnh giả, KHÔNG cần GPU/torch."""
import os
import sys
import tempfile
import shutil

import pytest
from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.models import CharacterLayer, LayerPlan  # noqa: E402
from app.services.storytelling.studio.studio_pipeline import (  # noqa: E402
    MatteQualityError, StudioPipeline)
from app.services.storytelling.studio.matting import ChromaMatter  # noqa: E402
from app.services.storytelling.studio.background_renderer import BackgroundRenderer  # noqa: E402

GREEN = "#00B140"


def _bg_blue(prompt, size):
    return Image.new("RGB", size, (0, 0, 255))


def _char_red_on_green(_layer):
    """Nhân vật đỏ chiếm ~40% khung, còn lại là nền xanh (coverage trong ngưỡng key)."""
    img = Image.new("RGB", (100, 150), (0, 177, 64))       # nền xanh
    inner = Image.new("RGB", (50, 120), (255, 0, 0))        # thân đỏ, kéo xuống đáy
    img.paste(inner, (25, 30))
    return img


def _char_all_green(_layer):
    return Image.new("RGB", (100, 150), (0, 177, 64))       # toàn nền → bị bỏ


class _SyntheticFallbackMatter:
    def cutout(self, image):
        rgba = image.convert("RGBA")
        alpha = Image.new("L", image.size, 0)
        alpha.paste(255, (25, 30, 75, 145))
        rgba.putalpha(alpha)
        return rgba


class _CoverageMatter:
    def __init__(self, coverage):
        self.coverage = coverage

    def cutout(self, image):
        rgba = image.convert("RGBA")
        alpha = Image.new("L", image.size, 0)
        width = int(round(image.size[0] * self.coverage))
        alpha.paste(255, (0, 0, width, image.size[1]))
        rgba.putalpha(alpha)
        return rgba


def _plan():
    layer = CharacterLayer(slug="a", prompt="p", anchor_x="center",
                           anchor_y="bottom", scale=0.9, z_order=0)
    return LayerPlan(location_id="loc", background_prompt="bg", characters=[layer])


def test_render_plan_composites_character():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "frames", "scene_000.png")
        sp = StudioPipeline(ctx_mgr=None, context=None)
        sp.render_plan(
            _plan(), (120, 90), out,
            bg_render_fn=_bg_blue, char_render_fn=_char_red_on_green,
            matter=ChromaMatter(bg_color=GREEN),
            bg_renderer=BackgroundRenderer(os.path.join(tmp, "bg"), enabled=False),
            harmonize=False,
        )
        assert os.path.exists(out)
        frame = Image.open(out).convert("RGB")
        assert frame.size == (120, 90)
        # thân nhân vật (đỏ) đã được ghép vào giữa khung
        r, g, b = frame.getpixel((60, 60))
        assert r > 150 and g < 90 and b < 90
        # đỉnh khung vẫn là nền (xanh dương) vì lớp neo đáy
        assert frame.getpixel((60, 3))[2] > 150
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_plan_rejects_failed_key():
    """Lớp bị key hỏng phải báo lỗi để run_batch fallback classic."""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "frames", "scene_000.png")
        sp = StudioPipeline(ctx_mgr=None, context=None)
        with pytest.raises(MatteQualityError, match="matte_quality:a"):
            sp.render_plan(
                _plan(), (120, 90), out,
                bg_render_fn=_bg_blue, char_render_fn=_char_all_green,
                matter=ChromaMatter(bg_color=GREEN),
                bg_renderer=BackgroundRenderer(os.path.join(tmp, "bg"), enabled=False),
                harmonize=False,
            )
        assert not os.path.exists(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_plan_uses_adaptive_matter_before_classic_fallback():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "frames", "scene_000.png")
        StudioPipeline().render_plan(
            _plan(), (120, 90), out,
            bg_render_fn=_bg_blue, char_render_fn=_char_all_green,
            matter=ChromaMatter(bg_color=GREEN),
            fallback_matter=_SyntheticFallbackMatter(),
            bg_renderer=BackgroundRenderer(os.path.join(tmp, "bg"), enabled=False),
            harmonize=False,
        )

        assert os.path.exists(out)
        assert Image.open(out).getpixel((60, 60))[1] > 100
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("coverage", [0.04, 0.96])
def test_render_plan_rejects_near_empty_or_near_full_matte(coverage):
    tmp = tempfile.mkdtemp()
    try:
        with pytest.raises(MatteQualityError):
            StudioPipeline().render_plan(
                _plan(), (120, 90), os.path.join(tmp, "frame.png"),
                bg_render_fn=_bg_blue, char_render_fn=_char_red_on_green,
                matter=_CoverageMatter(coverage),
                bg_renderer=BackgroundRenderer(os.path.join(tmp, "bg"), enabled=False),
                harmonize=False,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
