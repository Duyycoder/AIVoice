import os
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services.storytelling.models import Scene
from app.services.storytelling.studio import studio_pipeline as studio_mod
from app.services.storytelling.studio.studio_pipeline import StudioPipeline


class _Context:
    def __init__(self, characters):
        self.characters = characters

    @staticmethod
    def get_negative_prompt():
        return "bad quality"

    @staticmethod
    def get_positive_prompt():
        return "anime ink style, scenery background, detailed environment"


class _CtxMgr:
    def __init__(self, context, context_dir):
        self.context = context
        self.context_dir = str(context_dir)

    def get_character(self, value):
        norm = StudioPipeline._norm(value)
        return next((c for c in self.context.characters
                     if norm in (StudioPipeline._norm(c.name),
                                 StudioPipeline._norm(c.slug))), None)

    def load_context(self):
        return self.context

    @staticmethod
    def get_ref_image_path(_slug):
        return ""

    @staticmethod
    def get_face_embedding_path(_slug):
        return ""

    @staticmethod
    def has_identity(_slug):
        return False


class _FakePipe:
    def __init__(self, *, broken_matte=False, fail=False):
        self.broken_matte = broken_matte
        self.fail = fail
        self.events = []

    def warmup(self):
        self.events.append(("warmup", None))

    def update_ip_adapter_scale(self, value):
        self.events.append(("ip_scale", value))

    def set_character_lora(self, slug):
        self.events.append(("lora", slug))

    def generate_draft(self, **kwargs):
        prompt = kwargs["prompt"]
        self.events.append(("generate", prompt))
        if self.fail:
            raise RuntimeError("fake GPU failure")
        size = (kwargs["width"], kwargs["height"])
        if "flat green background" in prompt:
            img = Image.new("RGB", size, (0, 177, 64))
            if not self.broken_matte:
                w, h = size
                img.paste(Image.new("RGB", (w // 2, h * 3 // 4), (220, 20, 20)),
                          (w // 4, h // 4))
        else:
            img = Image.new("RGB", size, (20, 40, 180))
        return img, 123


def _character(name="Alice", slug="alice"):
    return SimpleNamespace(name=name, slug=slug, keywords_en="1girl, red robe")


def _scene(char_name="Alice"):
    scene = Scene(
        scene_id=0, text_vi="", word_count=0, start_time=0.0, end_time=1.0,
        duration_sec=1.0, image_prompt="1girl, red robe, ancient courtyard",
        characters_in_scene=[char_name], primary_character=char_name,
        fallback_level=0, accepted_seed=-1, frame_path="", shot_type="medium",
    )
    scene._semantic_meta = {"location": "ancient courtyard"}
    scene._llm_background_prompt = "empty ancient courtyard"
    return scene


def _cfg():
    return {
        "render_mode": "studio",
        "image_width": 120,
        "image_height": 80,
        "ip_adapter_scale": 0.6,
        "studio_matte_bg_color": "#00B140",
        "studio_matte_threshold": 0.18,
        "studio_matte_feather_px": 0,
        "studio_matte_despill": True,
        "studio_matte_adaptive_fallback": True,
        "studio_bg_cache": True,
        "studio_char_use_ip_adapter": False,
        "studio_char_use_detailer": False,
        "studio_layout_source": "heuristic",
        "studio_fallback_max_chars": 3,
        "studio_fallback_interaction_tags": [],
        "studio_shadow_opacity": 0.0,
        "enable_face_detailer": False,
    }


def _install_fake(monkeypatch, fake):
    import app.config as config_mod
    import app.services.storytelling.image_generator as image_generator
    monkeypatch.setattr(image_generator, "StorytellingPipeline", lambda _ctx: fake)
    monkeypatch.setattr(config_mod, "load_storytelling_config", _cfg)
    monkeypatch.setattr(studio_mod, "load_storytelling_config", _cfg)


def test_resolver_uses_word_boundaries_and_declared_order(tmp_path):
    chars = [_character("Lan", "lan"), _character("Binh", "binh")]
    pipeline = StudioPipeline(_CtxMgr(_Context(chars), tmp_path), _Context(chars))
    false_positive = SimpleNamespace(
        image_prompt="highly detailed landscape", characters_in_scene=[],
        primary_character="")
    ordered = SimpleNamespace(
        image_prompt="", characters_in_scene=["Binh", "Lan"],
        primary_character="")

    assert pipeline._resolve_slugs(false_positive) == []
    assert pipeline._resolve_slugs(ordered) == ["binh", "lan"]


def test_resolver_maps_primary_display_name_with_slug_only_context_manager(tmp_path):
    context = _Context([_character("Alice Display", "alice")])
    manager = _CtxMgr(context, tmp_path)
    manager.get_character = lambda value: next(
        (char for char in context.characters if char.slug == value), None)
    pipeline = StudioPipeline(manager, context)
    scene = SimpleNamespace(
        image_prompt="ancient courtyard", characters_in_scene=[],
        primary_character="Alice Display")

    assert pipeline._resolve_slugs(scene) == ["alice"]


@pytest.mark.parametrize("prompt", ["a young woman", "2girls", "village people"])
def test_person_prompt_without_resolved_character_requires_classic(prompt, tmp_path):
    pipeline = StudioPipeline(_CtxMgr(_Context([]), tmp_path), _Context([]))
    scene = SimpleNamespace(
        image_prompt=prompt, characters_in_scene=[], primary_character="")

    assert pipeline._unresolved_character_reason(scene, []) == (
        "person_prompt_without_resolved_character")


def test_plan_includes_action_and_story_style_and_sanitizes_background(tmp_path):
    context = _Context([_character()])
    pipeline = StudioPipeline(_CtxMgr(context, tmp_path), context)
    scene = _scene()
    scene.image_prompt = "1girl, holding a book, red robe, ancient courtyard"
    scene._semantic_meta = {
        "location": "ancient courtyard", "time_of_day": "sunset",
        "action": "đang cầm một quyển sách",
    }
    del scene._llm_background_prompt

    plan = pipeline.plan_scene(scene, ["alice"])

    assert plan.background_prompt.startswith(
        "anime ink style, scenery background, detailed environment, "
        "ancient courtyard, sunset")
    assert "1girl" not in plan.background_prompt
    # Hành động nằm ở `action` (renderer sẽ gắn trọng số), ngoại hình + style ở `prompt`.
    assert "holding a book" in plan.characters[0].action
    assert "anime ink style" in plan.characters[0].prompt
    assert "scenery background" not in plan.characters[0].prompt
    assert "detailed environment" not in plan.characters[0].prompt
    assert "đang cầm" not in plan.characters[0].prompt
    assert "đang cầm" not in plan.characters[0].action


def test_multi_character_uses_only_each_characters_llm_pose(monkeypatch, tmp_path):
    chars = [_character("Alice", "alice"), _character("Bob", "bob")]
    context = _Context(chars)
    pipeline = StudioPipeline(_CtxMgr(context, tmp_path), context)
    scene = _scene("Alice")
    scene.characters_in_scene = ["Alice", "Bob"]
    scene.image_prompt = "man holding a sword, woman watching, courtyard"
    scene._llm_layout = [
        {"name": "Alice", "anchor_x": 0.3, "anchor_y": 0.9,
         "scale": 0.7, "z": 0, "prompt": "holding a sword"},
        {"name": "Bob", "anchor_x": 0.7, "anchor_y": 0.9,
         "scale": 0.7, "z": 1, "prompt": "watching quietly"},
    ]
    monkeypatch.setattr(
        studio_mod, "load_storytelling_config",
        lambda: {"studio_layout_source": "llm"})

    plan = pipeline.plan_scene(scene, ["alice", "bob"])
    actions = {layer.slug: layer.action for layer in plan.characters}
    prompts = {layer.slug: layer.prompt for layer in plan.characters}

    assert "holding a sword" in actions["alice"]
    assert "watching quietly" not in actions["alice"]
    assert "watching quietly" in actions["bob"]
    assert "holding a sword" not in actions["bob"]
    # Ngoại hình không được nuốt pose của nhân vật khác
    assert "holding a sword" not in prompts["bob"]


def test_unresolved_declared_character_requires_classic(tmp_path):
    context = _Context([_character()])
    pipeline = StudioPipeline(_CtxMgr(context, tmp_path), context)

    reason = pipeline._unresolved_character_reason(_scene("Unknown"), [])

    assert reason.startswith("unresolved_characters:")


def test_run_batch_resets_lora_before_background(monkeypatch, tmp_path):
    context = _Context([_character()])
    fake = _FakePipe()
    _install_fake(monkeypatch, fake)
    out_dir = tmp_path / "draft_frames"

    result = StudioPipeline(_CtxMgr(context, tmp_path), context).run_batch(
        [_scene()], str(out_dir))

    assert os.path.exists(result[0].frame_path)
    bg_generate = next(i for i, event in enumerate(fake.events)
                       if event[0] == "generate" and "empty ancient courtyard" in event[1])
    assert fake.events[bg_generate - 1] == ("lora", None)
    assert ("lora", "alice") in fake.events[bg_generate + 1:]


def test_bad_matte_falls_back_to_classic(monkeypatch, tmp_path):
    context = _Context([_character()])
    fake = _FakePipe(broken_matte=True)
    _install_fake(monkeypatch, fake)

    result = StudioPipeline(_CtxMgr(context, tmp_path), context).run_batch(
        [_scene()], str(tmp_path / "draft_frames"))

    generated = [value for kind, value in fake.events if kind == "generate"]
    assert generated[-1] == _scene().image_prompt
    assert Image.open(result[0].frame_path).getpixel((0, 0)) == (20, 40, 180)


def test_double_render_failure_is_not_hidden_by_gray_frame(monkeypatch, tmp_path):
    context = _Context([_character()])
    fake = _FakePipe(fail=True)
    _install_fake(monkeypatch, fake)
    out_dir = tmp_path / "draft_frames"

    with pytest.raises(RuntimeError, match="Studio và fallback classic đều lỗi"):
        StudioPipeline(_CtxMgr(context, tmp_path), context).run_batch(
            [_scene()], str(out_dir))

    assert not (out_dir / "scene_000.png").exists()


def test_orchestrator_step2_runs_studio_and_persists_state(monkeypatch, tmp_path):
    from app.services.storytelling.orchestrator import StorytellingOrchestrator

    context = _Context([_character()])
    ctx_mgr = _CtxMgr(context, tmp_path / "context")
    fake = _FakePipe()
    _install_fake(monkeypatch, fake)
    orchestrator = StorytellingOrchestrator(ctx_mgr)
    orchestrator._state_path_override = str(tmp_path / "state.json")
    scene = _scene()
    scene._llm_layout = [{
        "name": "Alice", "anchor_x": 0.5, "anchor_y": 0.9,
        "scale": 0.8, "z": 0, "prompt": "holding a book",
    }]
    orchestrator.save_state("SCRIPT_READY", [scene], str(tmp_path))

    result = orchestrator.step2_generate_images([scene], str(tmp_path))

    with open(tmp_path / "state.json", "r", encoding="utf-8") as handle:
        state = json.load(handle)
    assert state["step"] == "STORYBOARD_READY"
    assert state["scenes"][0]["_semantic_meta"]["location"] == "ancient courtyard"
    assert state["scenes"][0]["_llm_layout"][0]["prompt"] == "holding a book"
    assert os.path.exists(result[0].frame_path)
