from app.services.storytelling.models import Scene, scene_from_dict, scene_to_dict


def test_studio_defaults_to_llm_layout_with_heuristic_fallback(monkeypatch):
    from app.config import Config

    monkeypatch.setattr(Config, "load_config", lambda self: None)

    assert Config().storytelling["studio_layout_source"] == "llm"


def test_scene_state_roundtrip_keeps_studio_metadata():
    scene = Scene(
        scene_id=1,
        text_vi="Cảnh thử",
        word_count=2,
        start_time=0.0,
        end_time=1.0,
        duration_sec=1.0,
        image_prompt="1boy, courtyard",
        characters_in_scene=["Dịch Phong"],
        primary_character="Dịch Phong",
        fallback_level=0,
        accepted_seed=-1,
        frame_path="",
        shot_type="medium",
    )
    scene._semantic_meta = {"location": "courtyard", "action": "standing"}
    scene._llm_background_prompt = "empty ancient courtyard"
    scene._llm_layout = [
        {"name": "Dịch Phong", "anchor_x": 0.5, "anchor_y": 0.9,
         "scale": 0.8, "z": 0}
    ]

    restored = scene_from_dict(scene_to_dict(scene))

    assert restored._semantic_meta == scene._semantic_meta
    assert restored._llm_background_prompt == scene._llm_background_prompt
    assert restored._llm_layout == scene._llm_layout


def test_scene_from_old_state_without_studio_metadata():
    raw = scene_to_dict(Scene(
        scene_id=0, text_vi="", word_count=0, start_time=0.0, end_time=0.0,
        duration_sec=0.0, image_prompt="scenery", characters_in_scene=[],
        primary_character="", fallback_level=0, accepted_seed=-1,
        frame_path="",
    ))

    restored = scene_from_dict(raw)

    assert not hasattr(restored, "_llm_layout")
