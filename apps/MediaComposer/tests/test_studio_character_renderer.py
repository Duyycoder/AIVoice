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
    # Không có hành động → rơi về tư thế mặc định, vẫn không có dấu phẩy thừa.
    assert build_character_prompt("", "green").startswith("standing, relaxed pose, solo")


def test_action_leads_prompt_with_weight():
    prompt = build_character_prompt(
        "black robe, silver hair", "gray", framing="full",
        action="swinging a sword downward, lunging forward")

    assert prompt.startswith("(swinging a sword downward:1.35), (lunging forward:1.35)")
    assert prompt.index("swinging a sword") < prompt.index("black robe")
    assert "standing, relaxed pose" not in prompt


def test_no_tag_forces_static_front_facing_pose():
    """Ba tag này từng triệt tiêu mọi hành động — không được quay lại."""
    prompt = build_character_prompt("black robe", "gray", action="running")

    assert "front view" not in prompt
    assert "centered subject" not in prompt
    assert "standing" not in prompt


def test_framing_tags_still_applied():
    assert "upper body" in build_character_prompt("x", "gray", framing="close")
    assert "cowboy shot" in build_character_prompt("x", "gray", framing="medium")
    assert "full body" in build_character_prompt("x", "gray", framing="full")


def test_style_tags_stripped_from_action():
    prompt = build_character_prompt(
        "black robe", "gray", action="masterpiece, running, cinematic lighting")
    assert "masterpiece" not in prompt
    assert "(running:1.35)" in prompt


def test_close_and_medium_negative_prompt_allows_intentional_crop():
    negative = "low quality, cropped, out of frame, cut off limbs"
    medium = build_character_negative_prompt(negative, "medium")
    full = build_character_negative_prompt(negative, "full")

    assert "low quality" in medium
    assert "cropped" not in medium
    assert "out of frame" not in medium
    assert "cut off" not in medium
    assert "cropped" in full
