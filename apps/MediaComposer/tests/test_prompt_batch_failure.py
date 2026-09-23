# -*- coding: utf-8 -*-
"""LLM chết sạch thì phải dừng, không được báo "Kịch bản đã sẵn sàng".

Bịt lỗ đã đốt hơn nửa tiếng GPU: `generate_prompts_batch` vứt hết kết quả của
từng cảnh (`for future in as_completed(futures): pass`). Khi mọi lời gọi LLM
hỏng, cả 62 cảnh rơi về cùng một prompt dự phòng chỉ-có-style, pipeline vẫn báo
thành công và render ra 62 tấm ảnh giống nhau, không dính gì tới chương truyện.
"""
import os
import sys

import pytest

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

import app.services.storytelling.llm_prompter as lp  # noqa: E402
from app.services.storytelling.models import Scene, StoryContext  # noqa: E402


def _scene(i):
    return Scene(
        scene_id=i, text_vi=f"Canh thu {i}", word_count=3,
        start_time=float(i), end_time=float(i + 1), duration_sec=1.0,
        image_prompt="", characters_in_scene=[], primary_character="",
        fallback_level=0, accepted_seed=-1, frame_path="",
    )


def _context():
    return StoryContext(story_name="Thu", story_slug="thu", genre="tien_hiep")


def test_moi_canh_hong_thi_nem_loi(monkeypatch):
    monkeypatch.setattr(lp, "_call_llm", lambda *a, **k: "")

    scenes = [_scene(i) for i in range(5)]
    with pytest.raises(lp.AllPromptsFailedError) as e:
        lp.generate_prompts_batch(scenes, _context())

    assert "5" in str(e.value), "thông báo phải nói rõ bao nhiêu cảnh hỏng"


def test_mot_so_canh_hong_van_chay_tiep(monkeypatch):
    """Hỏng một phần là chuyện bình thường — chỉ cảnh báo, không được dừng."""
    def gia_lap(messages, max_tokens=800):
        noi_dung = messages[-1]["content"]
        if not noi_dung.startswith("Scene text:"):
            return ""  # lời gọi Story Director
        # Ba cảnh đầu trả về được, ba cảnh sau hỏng — không phụ thuộc thứ tự luồng.
        for i in (0, 1, 2):
            if f"Canh thu {i}" in noi_dung:
                return '{"image_prompt": "1boy, standing in bamboo forest"}'
        return ""

    monkeypatch.setattr(lp, "_call_llm", gia_lap)

    scenes = [_scene(i) for i in range(6)]
    ket_qua = lp.generate_prompts_batch(scenes, _context())

    assert len(ket_qua) == 6
    assert any("bamboo forest" in s.image_prompt for s in ket_qua), \
        "cảnh thành công phải giữ được nội dung từ LLM"


def test_canh_da_co_prompt_thi_khong_tinh_la_hong(monkeypatch):
    """Chạy lại (resume) không được hiểu nhầm thành LLM chết."""
    monkeypatch.setattr(lp, "_call_llm", lambda *a, **k: "")

    scenes = [_scene(i) for i in range(4)]
    for s in scenes:
        s.image_prompt = "prompt da sinh tu lan chay truoc"

    ket_qua = lp.generate_prompts_batch(scenes, _context())

    assert len(ket_qua) == 4
