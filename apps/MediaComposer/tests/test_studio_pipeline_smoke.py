# -*- coding: utf-8 -*-
"""Smoke test StudioPipeline.render_plan — dùng render_fn giả, KHÔNG đụng Stable Diffusion.

studio_pipeline import app.config (kéo torch) → importorskip torch để bỏ qua nếu
môi trường không có torch. Các test thuần khác (matting/compositor/layout/bg_cache)
không cần torch và luôn chạy.
"""
import os
import sys
import tempfile
import shutil

import pytest
from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

pytest.importorskip("torch")  # studio_pipeline → app.config → torch

from app.services.storytelling.models import CharacterLayer, LayerPlan  # noqa: E402
from app.services.storytelling.studio.studio_pipeline import StudioPipeline  # noqa: E402
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


def test_render_plan_drops_failed_key():
    """Lớp bị key hỏng (toàn nền) → bỏ, frame chỉ còn nền."""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "frames", "scene_000.png")
        sp = StudioPipeline(ctx_mgr=None, context=None)
        sp.render_plan(
            _plan(), (120, 90), out,
            bg_render_fn=_bg_blue, char_render_fn=_char_all_green,
            matter=ChromaMatter(bg_color=GREEN),
            bg_renderer=BackgroundRenderer(os.path.join(tmp, "bg"), enabled=False),
            harmonize=False,
        )
        frame = Image.open(out).convert("RGB")
        r, g, b = frame.getpixel((60, 45))
        assert b > 150 and r < 90  # chỉ còn nền xanh dương
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
