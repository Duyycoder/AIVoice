# -*- coding: utf-8 -*-
"""Test layout heuristic + quyết định fallback classic — thuần, không cần GPU."""
import os
import sys

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.studio.layout_planner import (  # noqa: E402
    build_layer_plan, needs_classic_fallback, _distribute_anchor_x)


def test_shot_type_defaults():
    p_close = build_layer_plan("close", [{"slug": "a", "prompt": "x"}], "bg", "loc")
    assert p_close.characters[0].scale == 1.0
    assert p_close.characters[0].anchor_y == "middle"

    p_wide = build_layer_plan("wide", [{"slug": "a", "prompt": "x"}], "bg", "loc")
    assert p_wide.characters[0].scale == 0.45
    assert p_wide.characters[0].anchor_y == "bottom"

    # shot_type lạ → mặc định wide
    p_unk = build_layer_plan("weird", [{"slug": "a", "prompt": "x"}], "bg", "loc")
    assert p_unk.characters[0].scale == 0.45


def test_anchor_distribution():
    assert _distribute_anchor_x(1) == ["center"]
    assert _distribute_anchor_x(2) == ["left", "right"]
    assert _distribute_anchor_x(3) == ["left", "center", "right"]
    assert _distribute_anchor_x(0) == []


def test_layer_plan_fields():
    chars = [{"slug": "a", "prompt": "pa"}, {"slug": "b", "prompt": "pb"}]
    plan = build_layer_plan("medium", chars, "empty street", "street_01")
    assert plan.location_id == "street_01"
    assert plan.background_prompt == "empty street"
    assert plan.render_mode == "studio"
    assert [c.z_order for c in plan.characters] == [0, 1]
    assert [c.slug for c in plan.characters] == ["a", "b"]


def test_fallback_too_many_chars():
    assert needs_classic_fallback(4, "prompt", 3, []) is not None
    assert needs_classic_fallback(3, "prompt", 3, []) is None


def test_fallback_interaction_tag():
    tags = ["hug", "fight"]
    assert needs_classic_fallback(2, "two people in a hug scene", 3, tags) == "interaction_tag:hug"
    assert needs_classic_fallback(2, "calm standing scene", 3, tags) is None
