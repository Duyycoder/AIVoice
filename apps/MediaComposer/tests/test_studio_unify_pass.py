# -*- coding: utf-8 -*-
"""Test unify pass — phần thuần (clamp, prompt, resolve config, gọi có điều kiện)."""
import os
import sys

from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.models import CharacterLayer, LayerPlan  # noqa: E402
from app.services.storytelling.studio.studio_pipeline import StudioPipeline  # noqa: E402
from app.services.storytelling.studio.unify_pass import (  # noqa: E402
    MAX_SAFE_STRENGTH, build_unify_prompt, clamp_strength, resolve_unify_settings)


class _StubMatter:
    def cutout(self, img):
        rgba = img.convert("RGBA")
        # Giữ nửa khung → alpha coverage 0.5, qua cổng chất lượng
        px = rgba.load()
        for y in range(rgba.size[1] // 2, rgba.size[1]):
            for x in range(rgba.size[0]):
                px[x, y] = (0, 0, 0, 0)
        return rgba


class _StubBgRenderer:
    def get_or_render(self, location_id, prompt, size, render_fn):
        return render_fn(prompt, size)


def test_clamp_strength_blocks_composition_destroying_values():
    assert clamp_strength(0.28) == 0.28
    assert clamp_strength(0.9) == MAX_SAFE_STRENGTH
    assert clamp_strength(0) == 0.0
    assert clamp_strength(-1) == 0.0
    assert clamp_strength("bad") == 0.0


def test_resolve_unify_settings_off_switch():
    assert resolve_unify_settings({"studio_unify_pass": False}) is None
    assert resolve_unify_settings({"studio_unify_strength": 0}) is None

    cfg = resolve_unify_settings({"studio_unify_strength": 0.3, "studio_unify_steps": 20})
    assert cfg["strength"] == 0.3
    assert cfg["num_steps"] == 20


def test_unify_prompt_has_style_and_scene_but_no_character():
    prompt = build_unify_prompt(
        "flat vector illustration, minimal shading",
        "masterpiece, ancient street, red lanterns, no humans")

    assert prompt.startswith("flat vector illustration")
    assert "ancient street" in prompt
    assert "coherent lighting" in prompt
    # Tag chất lượng do prompt nền mang theo bị lọc để khỏi đánh nhau với style
    assert "masterpiece" not in prompt


def test_render_plan_runs_unify_only_when_layers_exist(tmp_path):
    pipeline = StudioPipeline()
    calls = []

    def unify_fn(frame, bg_prompt):
        calls.append(bg_prompt)
        return frame

    def bg_render_fn(prompt, size):
        return Image.new("RGB", size, (10, 20, 30))

    def char_render_fn(layer):
        return Image.new("RGB", (64, 96), (200, 100, 50))

    size = (128, 72)

    plan_bg_only = LayerPlan(location_id="loc", background_prompt="empty street",
                             characters=[])
    pipeline.render_plan(plan_bg_only, size, str(tmp_path / "a.png"),
                         bg_render_fn=bg_render_fn, char_render_fn=char_render_fn,
                         matter=_StubMatter(), bg_renderer=_StubBgRenderer(),
                         unify_fn=unify_fn)
    assert calls == [], "cảnh toàn nền không cần hòa trộn"

    plan_with_char = LayerPlan(
        location_id="loc", background_prompt="empty street",
        characters=[CharacterLayer(slug="a", prompt="p", action="running")])
    pipeline.render_plan(plan_with_char, size, str(tmp_path / "b.png"),
                         bg_render_fn=bg_render_fn, char_render_fn=char_render_fn,
                         matter=_StubMatter(), bg_renderer=_StubBgRenderer(),
                         unify_fn=unify_fn)
    assert calls == ["empty street"]
    assert os.path.exists(tmp_path / "b.png")
