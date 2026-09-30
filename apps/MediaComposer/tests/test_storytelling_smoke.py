# -*- coding: utf-8 -*-
"""Smoke tests cho storytelling pipeline — chạy không cần GPU."""
import os
import sys
import json
import tempfile
import shutil
import pytest

# Thêm path để import được modules
_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)


class TestImports:
    """Kiểm tra import tất cả module storytelling."""

    def test_import_models(self):
        from app.services.storytelling.models import Scene, Character, StoryContext

    def test_import_context_manager(self):
        from app.services.storytelling.context_manager import ContextManager

    def test_import_dataset_collector(self):
        from app.services.storytelling.dataset_collector import maybe_collect

    def test_import_lora_trainer(self):
        from app.services.storytelling.lora_trainer import (
            build_instance_prompt, get_trainable_characters
        )

    def test_import_semantic_splitter(self):
        from app.services.storytelling.semantic_scene_splitter import (
            split_scenes_semantic, SemanticScene
        )

    def test_import_srt_mapper(self):
        from app.services.storytelling.srt_mapper import (
            map_scenes_to_timeline, map_semantic_scenes_to_srt
        )

    def test_import_batch_video_runner(self):
        from app.services.storytelling.batch_video_runner import (
            scan_batch_dir, BatchItem, BatchReport
        )


class TestSemanticSplitter:
    """Tách cảnh: LLM chỉ đánh dấu mốc, code dựng + cân độ dài (không cần LLM thật)."""

    def test_extract_paragraphs(self):
        from app.services.storytelling.semantic_scene_splitter import _extract_paragraphs
        md = "# Title\n\nĐoạn một dài đủ năm từ.\n\nĐoạn hai dài đủ năm từ.\n\nhttp://link.com\n"
        result = _extract_paragraphs(md)
        assert len(result) == 2
        assert "Title" not in result[0]
        assert "http" not in result[1]

    def test_boundaries_drop_invalid_and_cover_all(self):
        from app.services.storytelling.semantic_scene_splitter import _boundaries_to_ranges
        # Bỏ đoạn 0, trùng mốc, vượt chương (70 trên chương 10 đoạn) — đúng kiểu lỗi LLM thật
        raw = [{"start": 3, "location": "sân"}, {"start": 3}, {"start": 7}, {"start": 70}]
        ranges = _boundaries_to_ranges(raw, 10)
        assert [(r["start"], r["end"]) for r in ranges] == [(0, 3), (3, 7), (7, 10)]
        assert ranges[1]["location"] == "sân"

    def test_normalize_merges_same_location(self):
        from app.services.storytelling.semantic_scene_splitter import (
            _boundaries_to_ranges, _normalize_ranges)
        words = [10] * 6
        raw = [{"start": 0, "location": "Sân", "time_of_day": "day"},
               {"start": 2, "location": "sân", "time_of_day": "day"},
               {"start": 4, "location": "Hậu viện", "time_of_day": "day"}]
        ranges = _normalize_ranges(_boundaries_to_ranges(raw, 6), words, 60.0)
        assert [(r["start"], r["end"]) for r in ranges] == [(0, 4), (4, 6)]

    def test_normalize_caps_scene_count(self):
        from app.services.storytelling.semantic_scene_splitter import (
            _boundaries_to_ranges, _normalize_ranges)
        # 65 đoạn, mỗi đoạn là 1 mốc (LLM hỏng -> chia bằng code), chương 465s
        n = 65
        ranges = _normalize_ranges(_boundaries_to_ranges([{"start": i} for i in range(n)], n),
                                   [23] * n, 465.0)
        assert 8 <= len(ranges) <= 16
        assert ranges[0]["start"] == 0 and ranges[-1]["end"] == n
        assert all(a["end"] == b["start"] for a, b in zip(ranges, ranges[1:]))

    def test_normalize_splits_long_scene(self):
        from app.services.storytelling.semantic_scene_splitter import (
            _boundaries_to_ranges, _normalize_ranges)
        # 1 cảnh 200s -> phải bị cắt thành các cảnh <= 60s
        ranges = _normalize_ranges(_boundaries_to_ranges([{"start": 0}], 10), [10] * 10, 200.0)
        assert len(ranges) >= 4
        assert all((r["end"] - r["start"]) * 20.0 <= 60.0 for r in ranges)

    def test_split_uses_llm_boundaries(self, monkeypatch):
        from app.services.storytelling import semantic_scene_splitter as s
        md = "\n\n".join(f"Đoạn số {i} kể chuyện đủ dài năm từ trở lên." for i in range(12))
        monkeypatch.setattr(s, "_call_llm_boundaries", lambda paras, dur: [
            {"start": 0, "location": "sân", "characters": ["A"], "action": "chào"},
            {"start": 6, "location": "rừng", "characters": ["B"], "action": "chạy"}])
        scenes = s.split_scenes_semantic(md, 80.0)
        assert [sc.paragraph_indices[0] for sc in scenes] == [0, 6]
        assert scenes[1].location == "rừng" and scenes[1].characters == ["B"]

    def test_split_without_llm_still_groups(self, monkeypatch):
        import app.services.llm as llm
        from app.services.storytelling import semantic_scene_splitter as s

        def boom():
            raise RuntimeError("no llm")
        monkeypatch.setattr(llm, "get_llm_client", boom)
        md = "\n\n".join(f"Đoạn số {i} kể chuyện đủ dài năm từ trở lên nữa." for i in range(40))
        scenes = s.split_scenes_semantic(md, 400.0)
        assert scenes and 5 <= len(scenes) <= 14

    def test_director_groups_fit_context(self):
        from app.services.storytelling.llm_prompter import (
            _director_groups, DIRECTOR_GROUP_MAX_SCENES, DIRECTOR_GROUP_WORDS)
        from app.services.storytelling.models import Scene
        scenes = [Scene(scene_id=i, text_vi=" ".join(["từ"] * 40), word_count=40,
                        start_time=0, end_time=0, duration_sec=0, image_prompt="",
                        characters_in_scene=[], primary_character="", fallback_level=0,
                        accepted_seed=-1, frame_path="") for i in range(52)]
        groups = _director_groups(scenes)
        assert sum(len(g) for g in groups) == 52
        assert all(len(g) <= DIRECTOR_GROUP_MAX_SCENES for g in groups)
        assert all(sum(40 for _ in g) <= DIRECTOR_GROUP_WORDS for g in groups)


class TestSRTMapper:
    """Test map_semantic_scenes_to_srt."""

    def test_no_srt_blocks_returns_proportional(self):
        from app.services.storytelling.models import Scene
        from app.services.storytelling.srt_mapper import map_semantic_scenes_to_srt
        scenes = [
            Scene(scene_id=0, text_vi="hello world test", word_count=3,
                  start_time=0, end_time=0, duration_sec=0,
                  image_prompt="", characters_in_scene=[], primary_character="",
                  fallback_level=0, accepted_seed=-1, frame_path=""),
            Scene(scene_id=1, text_vi="second scene here", word_count=3,
                  start_time=0, end_time=0, duration_sec=0,
                  image_prompt="", characters_in_scene=[], primary_character="",
                  fallback_level=0, accepted_seed=-1, frame_path=""),
        ]
        result = map_semantic_scenes_to_srt(scenes, [], 10.0)
        assert len(result) == 2
        # Cảnh cuối kết thúc tại total_duration
        assert result[-1].end_time == 10.0

    def test_no_srt_blocks_weights_timing_by_word_count(self):
        from app.services.storytelling.models import Scene
        from app.services.storytelling.srt_mapper import map_semantic_scenes_to_srt
        scenes = [
            Scene(scene_id=0, text_vi="ngắn", word_count=1,
                  start_time=0, end_time=0, duration_sec=0,
                  image_prompt="", characters_in_scene=[], primary_character="",
                  fallback_level=0, accepted_seed=-1, frame_path=""),
            Scene(scene_id=1, text_vi="đoạn dài ba từ", word_count=4,
                  start_time=0, end_time=0, duration_sec=0,
                  image_prompt="", characters_in_scene=[], primary_character="",
                  fallback_level=0, accepted_seed=-1, frame_path=""),
        ]

        result = map_semantic_scenes_to_srt(scenes, [], 10.0)

        assert result[0].start_time == 0.0
        assert result[0].end_time == pytest.approx(2.0)
        assert result[1].start_time == pytest.approx(2.0)
        assert result[1].end_time == 10.0


class TestDatasetFIFO:
    """Test FIFO eviction trong add_dataset_image."""

    def test_fifo_eviction_auto_images(self):
        from app.services.storytelling.context_manager import ContextManager
        from PIL import Image as PILImage

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup fake context dir
            chars_dir = os.path.join(tmpdir, "characters")
            ds_dir = os.path.join(chars_dir, "test_char", "dataset")
            os.makedirs(ds_dir)

            # Tạo ContextManager giả
            ctx_mgr = object.__new__(ContextManager)
            ctx_mgr.chars_dir = chars_dir

            # Tạo 40 ảnh auto giả
            for i in range(40):
                fake_path = os.path.join(ds_dir, f"auto_{i:08x}.png")
                PILImage.new("RGB", (10, 10)).save(fake_path)

            assert ctx_mgr.count_dataset_images("test_char") == 40

            # Thêm 1 ảnh nữa → phải evict 1 auto cũ nhất
            new_img = PILImage.new("RGB", (10, 10))
            ctx_mgr.add_dataset_image("test_char", new_img, "auto")

            assert ctx_mgr.count_dataset_images("test_char") == 40  # không vượt 40


class TestLoadContextBackwardCompat:
    """Test load context.json cũ không crash."""

    def test_load_old_context_no_lora_fields(self):
        from app.services.storytelling.models import Character
        import dataclasses

        # Giả lập context.json cũ (thiếu lora_status, auto_collect, etc.)
        old_data = {
            "name": "Old Char",
            "slug": "old_char",
            "description": "desc",
            "keywords_en": "tag1, tag2",
            "has_embedding": False,
        }
        valid_fields = {f.name for f in dataclasses.fields(Character)}
        filtered = {k: v for k, v in old_data.items() if k in valid_fields}
        char = Character(**filtered)

        assert char.lora_status == "none"
        assert char.auto_collect is True

    def test_load_context_with_unknown_keys(self):
        from app.services.storytelling.models import Character
        import dataclasses

        # Context.json có key lạ từ tương lai
        future_data = {
            "name": "Future Char",
            "slug": "future_char",
            "description": "desc",
            "keywords_en": "tag1",
            "has_embedding": True,
            "lora_status": "trained",
            "future_field": "should_be_filtered",
        }
        valid_fields = {f.name for f in dataclasses.fields(Character)}
        filtered = {k: v for k, v in future_data.items() if k in valid_fields}
        char = Character(**filtered)

        assert char.lora_status == "trained"
        assert not hasattr(char, "future_field")


class TestLoRATrainer:
    """Test build_instance_prompt."""

    def test_build_prompt_with_instance(self):
        from app.services.storytelling.lora_trainer import build_instance_prompt
        from app.services.storytelling.models import Character

        char = Character(
            name="Test", slug="test", description="", keywords_en="",
            has_embedding=False, instance_prompt="custom prompt"
        )
        assert build_instance_prompt(char) == "custom prompt"

    def test_build_prompt_from_keywords(self):
        from app.services.storytelling.lora_trainer import build_instance_prompt
        from app.services.storytelling.models import Character

        char = Character(
            name="Test", slug="test", description="",
            keywords_en="black hair, blue eyes, upper body, looking at viewer, tall",
            has_embedding=False,
        )
        result = build_instance_prompt(char)
        assert "upper body" not in result
        assert "looking at viewer" not in result
        assert "black hair" in result
        assert "blue eyes" in result


class TestBatchVideoRunner:
    """Test scan_batch_dir."""

    def test_scan_flat_layout(self):
        from app.services.storytelling.batch_video_runner import scan_batch_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            # Tạo 2 cặp file
            for stem in ["ep01", "ep02"]:
                open(os.path.join(tmpdir, f"{stem}.md"), "w").close()
                open(os.path.join(tmpdir, f"{stem}.wav"), "w").close()
            # Tạo 1 file orphan (chỉ md, không audio)
            open(os.path.join(tmpdir, "orphan.md"), "w").close()

            items = scan_batch_dir(tmpdir)
            assert len(items) == 2
            assert items[0].stem == "ep01"
            assert items[1].stem == "ep02"

    def test_scan_subdir_layout(self):
        from app.services.storytelling.batch_video_runner import scan_batch_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["chapter1", "chapter2"]:
                subdir = os.path.join(tmpdir, name)
                os.makedirs(subdir)
                open(os.path.join(subdir, "script.md"), "w").close()
                open(os.path.join(subdir, "audio.mp3"), "w").close()

            items = scan_batch_dir(tmpdir)
            assert len(items) == 2
