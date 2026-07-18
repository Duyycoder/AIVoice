from app.services.storytelling.studio.layout_planner import build_layer_plan_from_llm

def test_build_from_llm_valid():
    llm_layout = [
        {"name": "Dịch Phong", "anchor_x": 0.5, "anchor_y": 0.9, "scale": 0.8, "z": 0},
        {"name": "Tiểu muội", "anchor_x": 0.8, "anchor_y": 0.8, "scale": 0.7, "z": 1}
    ]
    chars = [
        {"slug": "dich_phong", "prompt": "1boy, handsome"},
        {"slug": "tieu_muoi", "prompt": "1girl, cute"}
    ]
    
    plan = build_layer_plan_from_llm(llm_layout, chars, "clean room", "loc_123")
    assert plan is not None
    assert plan.location_id == "loc_123"
    assert plan.background_prompt == "clean room"
    assert len(plan.characters) == 2
    
    # Check layer 1
    assert plan.characters[0].slug == "dich_phong"
    assert plan.characters[0].anchor_x == 0.5
    assert plan.characters[0].anchor_y == 0.9
    assert plan.characters[0].scale == 0.8
    assert plan.characters[0].z_order == 0
    
    # Check layer 2
    assert plan.characters[1].slug == "tieu_muoi"
    assert plan.characters[1].anchor_x == 0.8
    assert plan.characters[1].z_order == 1

def test_build_from_llm_adds_character_pose_prompt():
    layout = [{"name": "A", "anchor_x": 0.5, "anchor_y": 0.9,
               "scale": 0.8, "z": 0, "prompt": "holding a sword, angry"}]
    chars = [{"name": "A", "slug": "a", "prompt": "black robe"}]

    plan = build_layer_plan_from_llm(layout, chars, "", "", shot_type="close")

    assert plan.characters[0].prompt == "holding a sword, angry, black robe"
    assert plan.characters[0].framing == "close"

def test_build_from_llm_invalid_falls_none():
    # Thiếu dictionary
    assert build_layer_plan_from_llm([["invalid"]], [{"slug": "a", "prompt": ""}], "", "") is None
    
    # Anchor ngoài phạm vi
    layout_out_of_bounds = [{"name": "A", "anchor_x": 1.5, "anchor_y": 0.5, "scale": 0.5, "z": 0}]
    assert build_layer_plan_from_llm(layout_out_of_bounds, [{"slug": "a", "prompt": ""}], "", "") is None
    
    # Thiếu name -> bỏ qua
    layout_no_name = [{"anchor_x": 0.5, "anchor_y": 0.5, "scale": 0.5, "z": 0}]
    assert build_layer_plan_from_llm(layout_no_name, [{"slug": "a", "prompt": ""}], "", "") is None

def test_build_from_llm_name_normalization():
    layout = [{"name": "Dịch Phong", "anchor_x": 0.5, "anchor_y": 0.9, "scale": 0.8, "z": 0}]
    # Slug trong chars là "dichphong" -> norm("Dịch Phong") cũng là "dichphong"
    chars = [{"slug": "dichphong", "prompt": "..."}]
    
    plan = build_layer_plan_from_llm(layout, chars, "", "")
    assert plan is not None
    assert plan.characters[0].slug == "dichphong"

def test_build_from_llm_maps_display_name_to_legacy_slug():
    layout = [{"name": "Dịch Phong", "anchor_x": 0.5, "anchor_y": 0.9, "scale": 0.8, "z": 0}]
    chars = [{"name": "Dịch Phong", "slug": "d_ch_phong", "prompt": "..."}]

    plan = build_layer_plan_from_llm(layout, chars, "", "")
    assert plan is not None
    assert plan.characters[0].slug == "d_ch_phong"

def test_build_from_llm_missing_character_falls_none():
    layout = [{"name": "A", "anchor_x": 0.25, "anchor_y": 0.9, "scale": 0.8, "z": 0}]
    chars = [
        {"name": "A", "slug": "a", "prompt": "..."},
        {"name": "B", "slug": "b", "prompt": "..."},
    ]

    assert build_layer_plan_from_llm(layout, chars, "", "") is None

class DummyScene:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_plan_scene_llm_source(monkeypatch):
    from app.services.storytelling.studio.studio_pipeline import StudioPipeline
    
    # Mock config
    def mock_cfg():
        return {"studio_layout_source": "llm"}
    monkeypatch.setattr("app.services.storytelling.studio.studio_pipeline.load_storytelling_config", mock_cfg)
    
    scene = DummyScene(
        image_prompt="A boy in room",
        characters_in_scene=["dichphong"],
        _llm_layout=[{"name": "Dịch Phong", "anchor_x": 0.5, "anchor_y": 0.9, "scale": 0.8, "z": 0}],
        _llm_background_prompt="clean room"
    )
    
    pipeline = StudioPipeline(ctx_mgr=None, context=None)
    
    # Mock appearance
    def mock_app(slug): return "prompt"
    pipeline._appearance_for = mock_app
    
    plan = pipeline.plan_scene(scene, ["dichphong"])
    assert plan.background_prompt == "clean room, no humans, no people, scenery, empty background"
    assert len(plan.characters) == 1
    assert plan.characters[0].anchor_x == 0.5

def test_plan_scene_llm_fallback(monkeypatch):
    from app.services.storytelling.studio.studio_pipeline import StudioPipeline
    def mock_cfg():
        return {"studio_layout_source": "llm"}
    monkeypatch.setattr("app.services.storytelling.studio.studio_pipeline.load_storytelling_config", mock_cfg)
    
    # Scene KHÔNG có _llm_layout
    scene = DummyScene(image_prompt="A boy in room", characters_in_scene=["dichphong"])
    
    pipeline = StudioPipeline(ctx_mgr=None, context=None)
    def mock_app(slug): return "prompt"
    pipeline._appearance_for = mock_app
    
    plan = pipeline.plan_scene(scene, ["dichphong"])
    # Do không có _llm_layout, fallback heuristic -> anchor_x là string "center"
    assert plan.characters[0].anchor_x == "center"

def test_prompter_parses_layout_fields(monkeypatch):
    from app.services.storytelling.llm_prompter import _process_scene_with_retry
    
    scene = DummyScene(image_prompt="", text_vi="")
    # Giả lập JSON LLM trả về
    data = {
        "background_prompt": "empty street",
        "layout": [{"name": "A", "anchor_x": 0.5, "anchor_y": 0.5, "scale": 0.8, "z": 0}]
    }
    
    # Mock LLM call
    def mock_call(*args, **kwargs):
        import json
        return json.dumps(data)
    monkeypatch.setattr("app.services.storytelling.llm_prompter._call_llm", mock_call)
        
    class DummyContext:
        checkpoint = "test"
    
    _process_scene_with_retry(scene, "", DummyContext(), 1, "")
    
    assert getattr(scene, "_llm_background_prompt", None) == "empty street"
    assert len(getattr(scene, "_llm_layout", [])) == 1

def test_prompter_missing_layout_fields(monkeypatch):
    from app.services.storytelling.llm_prompter import _process_scene_with_retry
    
    scene = DummyScene(image_prompt="", text_vi="")
    data = {"image_prompt": "test"} # Thiếu background_prompt và layout
    
    def mock_call(*args, **kwargs):
        import json
        return json.dumps(data)
    monkeypatch.setattr("app.services.storytelling.llm_prompter._call_llm", mock_call)
    
    class DummyContext:
        checkpoint = "test"
        
    _process_scene_with_retry(scene, "", DummyContext(), 1, "")
    
    assert getattr(scene, "_llm_background_prompt", "not_set") is None
    assert getattr(scene, "_llm_layout", "not_set") is None
