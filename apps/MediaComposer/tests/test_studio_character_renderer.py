from app.services.storytelling.studio.character_renderer import (
    bg_color_name,
    build_character_negative_prompt,
    build_character_prompt,
)


def test_custom_matte_color_uses_matching_color_family():
    assert bg_color_name("#ff0000") == "red"
    assert bg_color_name("#00ffff") == "cyan"
    assert bg_color_name("#ff00ff") == "magenta"


def test_character_prompt_has_no_leading_separator_when_appearance_empty():
    assert build_character_prompt("", "green").startswith("solo, 1 person")


def test_close_and_medium_negative_prompt_allows_intentional_crop():
    negative = "low quality, cropped, out of frame, cut off limbs"
    medium = build_character_negative_prompt(negative, "medium")
    full = build_character_negative_prompt(negative, "full")

    assert "low quality" in medium
    assert "cropped" not in medium
    assert "out of frame" not in medium
    assert "cut off" not in medium
    assert "cropped" in full
