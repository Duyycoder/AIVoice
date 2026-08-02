# -*- coding: utf-8 -*-
"""Test preset phong cách thật sự được áp vào truyện.

Bịt lại lỗ: `adapter_video_cli` từng chỉ gán `context.art_style = <tên>` — một
chuỗi không ai đọc — nên chọn phong cách trên UI không đổi được ảnh, mọi truyện
đều dùng style mặc định ghi lúc tạo context.
"""
import os
import sys

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

import app.services.storytelling.context_manager as cm_mod  # noqa: E402
from app.services.storytelling.context_manager import ContextManager  # noqa: E402

PRESETS_DIR = os.path.join(_MC_ROOT, "resource", "image_presets")


def _mgr(tmp_path, monkeypatch, slug="probe"):
    monkeypatch.setattr(cm_mod, "CONTEXTS_ROOT", str(tmp_path))
    return ContextManager(slug)


def test_default_preset_file_exists():
    """Mặc định phải trỏ tới một preset CÓ THẬT, nếu không truyện mới sẽ rơi
    về chuỗi dự phòng mà không ai biết."""
    path = os.path.join(PRESETS_DIR, f"{ContextManager.DEFAULT_STYLE_PRESET}.txt")
    assert os.path.exists(path), f"thiếu preset mặc định: {path}"


def test_apply_style_preset_writes_style_file(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch)
    assert mgr.apply_style_preset("thuy_mac") is True

    with open(mgr.style_file, encoding="utf-8") as f:
        content = f.read()
    with open(os.path.join(PRESETS_DIR, "thuy_mac.txt"), encoding="utf-8") as f:
        expected = f.read()
    assert content == expected
    assert "ink wash" in content
    assert "---" in content, "preset phải có cả phần negative"


def test_apply_style_preset_switches_between_styles(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch)
    mgr.apply_style_preset("thuy_mac")
    mgr.apply_style_preset("flat_anime")

    with open(mgr.style_file, encoding="utf-8") as f:
        content = f.read()
    assert "ink wash" not in content, "đổi style phải ghi đè, không phải nối thêm"


def test_unknown_preset_keeps_current_style(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch)
    mgr.apply_style_preset("thuy_mac")

    assert mgr.apply_style_preset("khong_ton_tai") is False
    with open(mgr.style_file, encoding="utf-8") as f:
        assert "ink wash" in f.read(), "preset sai không được xoá style đang dùng"


def test_empty_preset_name_is_noop(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch)
    assert mgr.apply_style_preset("") is False


def test_new_context_uses_default_preset(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, slug="fresh")
    ctx = mgr.create_context("Truyện mới", "tien_hiep")

    assert ctx.art_style == ContextManager.DEFAULT_STYLE_PRESET
    positive = ctx.get_positive_prompt()
    assert "ink wash" in positive
    # Style cũ (flat color anime) không được quay lại làm mặc định
    assert "minimalist anime" not in positive
    assert ctx.get_negative_prompt().strip(), "phải có negative prompt"


def test_ui_style_values_all_map_to_real_presets():
    """Mọi value trong dropdown phải trùng tên file preset.

    Dropdown Cài đặt từng dùng 'anime_2d_flat' / 'xianxia_cultivation' /
    'watercolor_storytelling' trong khi file thật tên 'flat_anime' / 'xianxia' /
    'watercolor' — chọn gì cũng không khớp.
    """
    import re
    html_path = os.path.abspath(os.path.join(
        _MC_ROOT, "..", "..", "..", "webui", "index.html"))
    if not os.path.exists(html_path):
        return  # submodule tách rời khỏi repo tổng
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    for select_id in ("s3Style", "cfgVideoStyle"):
        block = re.search(
            rf'<select id="{select_id}">(.*?)</select>', html, re.DOTALL)
        assert block, f"không tìm thấy <select id={select_id}>"
        values = re.findall(r'<option value="([^"]+)"', block.group(1))
        assert values, f"{select_id} không có option nào"
        for value in values:
            preset = os.path.join(PRESETS_DIR, f"{value}.txt")
            assert os.path.exists(preset), \
                f"{select_id} có value '{value}' nhưng không có {value}.txt"
