# -*- coding: utf-8 -*-
"""Test khóa style + đưa hành động lên đầu prompt — thuần, không cần GPU."""
import os
import sys

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.style_lock import (  # noqa: E402
    build_locked_prompt, dedupe_against, strip_style_drift, weight_action)

STYLE = ("(masterpiece, best quality:1.2), flat vector illustration, "
         "minimal shading, limited color palette")


def test_strip_style_drift_removes_quality_and_medium_tags():
    raw = ("masterpiece, best quality, highres, cinematic lighting, anime, "
           "ancient temple courtyard, falling snow, stone steps")
    out = strip_style_drift(raw)

    assert "masterpiece" not in out
    assert "best quality" not in out
    assert "cinematic lighting" not in out
    assert "anime" not in out
    # Nội dung cụ thể phải giữ nguyên
    assert "ancient temple courtyard" in out
    assert "falling snow" in out
    assert "stone steps" in out


def test_strip_style_drift_keeps_content_that_merely_contains_a_style_word():
    # "candlelight" chứa "light" nhưng là vật thể trong cảnh — không được cắt.
    out = strip_style_drift("candlelight, detailed, red lanterns")
    assert "candlelight" in out
    assert "red lanterns" in out
    assert "detailed" not in out.split(", ")


def test_strip_style_drift_removes_model_names():
    assert "Anything V5" not in strip_style_drift("Anything V5:1.1, wooden bridge")
    assert "wooden bridge" in strip_style_drift("Anything V5:1.1, wooden bridge")


def test_weight_action_wraps_each_tag():
    assert weight_action("running, holding a sword", 1.35) == \
        "(running:1.35), (holding a sword:1.35)"


def test_weight_action_preserves_existing_weights():
    assert weight_action("(kneeling:1.2)", 1.35) == "(kneeling:1.2)"


def test_dedupe_against_drops_tags_already_in_reference():
    assert dedupe_against("flat vector illustration, temple", STYLE) == "temple"


def test_locked_prompt_puts_style_first_then_weighted_action():
    out = build_locked_prompt(
        STYLE,
        "masterpiece, ancient temple courtyard, falling snow, wide shot",
        action="swinging a sword downward, lunging forward")

    assert out.startswith("(masterpiece, best quality:1.2), flat vector illustration")
    # Hành động phải đứng TRƯỚC nội dung — đây là điều model thực sự đọc.
    assert out.index("(swinging a sword downward:1.35)") < out.index("ancient temple courtyard")
    # Tag style do LLM tự thêm bị loại
    assert out.count("masterpiece") == 1
    assert "wide shot" in out


def test_locked_prompt_without_action_is_still_style_first():
    out = build_locked_prompt(STYLE, "empty street, red lanterns")
    assert out.startswith("(masterpiece, best quality:1.2)")
    assert "empty street" in out
    assert "(" not in out.split("flat vector illustration")[1].split(",")[0]


def test_locked_prompt_does_not_duplicate_action_in_content():
    out = build_locked_prompt(
        STYLE, "running, muddy road, night", action="running")
    assert out.count("running") == 1
    assert "(running:1.35)" in out
