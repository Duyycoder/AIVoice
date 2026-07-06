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
    """Test semantic scene splitter validation logic (không cần LLM)."""

    def test_validate_valid_scenes(self):
        from app.services.storytelling.semantic_scene_splitter import _validate_scene_coverage
        raw = [
            {"paragraphs": [0, 1, 2]},
            {"paragraphs": [3, 4]},
            {"paragraphs": [5]},
        ]
        assert _validate_scene_coverage(raw, 6) is True

    def test_validate_missing_paragraph(self):
        from app.services.storytelling.semantic_scene_splitter import _validate_scene_coverage
        raw = [
            {"paragraphs": [0, 1]},
            {"paragraphs": [3, 4]},  # bỏ sót [2]
        ]
        assert _validate_scene_coverage(raw, 5) is False

    def test_validate_overlapping(self):
        from app.services.storytelling.semantic_scene_splitter import _validate_scene_coverage
        raw = [
            {"paragraphs": [0, 1, 2]},
            {"paragraphs": [2, 3]},  # chồng lấn [2]
        ]
        assert _validate_scene_coverage(raw, 4) is False

    def test_validate_non_contiguous(self):
        from app.services.storytelling.semantic_scene_splitter import _validate_scene_coverage
        raw = [
            {"paragraphs": [0, 2]},  # không liên tiếp
            {"paragraphs": [1, 3]},
        ]
        assert _validate_scene_coverage(raw, 4) is False

    def test_extract_paragraphs(self):
        from app.services.storytelling.semantic_scene_splitter import _extract_paragraphs
        md = "# Title\n\nĐoạn một dài đủ năm từ.\n\nĐoạn hai dài đủ năm từ.\n\nhttp://link.com\n"
        result = _extract_paragraphs(md)
        assert len(result) == 2
        assert "Title" not in result[0]
        assert "http" not in result[1]

    def test_enforce_duration_merges_short(self):
        from app.services.storytelling.semantic_scene_splitter import (
            SemanticScene, _enforce_duration_bounds
        )
        # Cảnh 1: 50 từ (≈5s / 100s = quá ngắn < 8s)
        # Cảnh 2: 50 từ (≈5s / 100s = quá ngắn)
        # Cảnh 3: 900 từ (≈90s — dài)
        # Kết quả: cảnh 1 & 2 gộp thành 1 (10s OK), cảnh 3 giữ nguyên → 2 cảnh
        scenes = [
            SemanticScene(0, " ".join(["word"] * 50), "", "", [], "", "", [0]),
            SemanticScene(1, " ".join(["word"] * 50), "", "", [], "", "act2", [1]),
            SemanticScene(2, " ".join(["word"] * 900), "", "", [], "", "act3", [2, 3]),
        ]
        result = _enforce_duration_bounds(scenes, 1000, 100.0, 8.0, 15.0)
        # Cảnh 1+2 gộp (10s OK), cảnh 3 giữ (90s > 27 nhưng chỉ có 2 paragraphs → cắt đôi)
        assert len(result) >= 2
        # Verify cảnh đầu tiên chứa "act2" (đã gộp)
        assert "act2" in result[0].action


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
