# -*- coding: utf-8 -*-
"""Override LLM đặt trong bộ nhớ phải sống sót qua mọi lần nạp lại config.

Bịt lỗ đã làm hỏng nguyên một lượt dựng video: `adapter_video_cli` gán thẳng
`config.app["llm_api_key"]`, nhưng `load_storytelling_config()` — được gọi ở 19
chỗ trong luồng dựng, kể cả giữa lúc sinh prompt — chạy `load_config()` và
`update()` đè giá trị rỗng của config.toml lên. Kết quả: key bị xoá giữa chừng,
toàn bộ 62 cảnh sinh prompt hỏng với "Chưa cấu hình API Key" mà pipeline vẫn báo
"Kịch bản đã sẵn sàng" rồi render tiếp.
"""
import os
import sys

import toml

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

import app.config as config_mod  # noqa: E402


def _cfg_tren_file_tam(tmp_path, app_section=None):
    """Config trỏ vào config.toml tạm, không đụng file thật của dự án."""
    cfg = config_mod.Config()
    cfg.config_file = str(tmp_path / "config.toml")
    with open(cfg.config_file, "w", encoding="utf-8") as f:
        toml.dump({"app": app_section if app_section is not None else {
            "llm_api_key": "", "llm_base_url": "", "llm_model": ""}}, f)
    cfg.load_config()
    return cfg


def test_override_song_sot_qua_load_config(tmp_path):
    cfg = _cfg_tren_file_tam(tmp_path)
    cfg.set_app_override("llm_api_key", "ollama")
    cfg.set_app_override("llm_model", "qwen2.5:3b-instruct")

    cfg.load_config()  # chính lời gọi đã xoá key giữa chừng

    assert cfg.app["llm_api_key"] == "ollama"
    assert cfg.app["llm_model"] == "qwen2.5:3b-instruct"


def test_load_config_lap_nhieu_lan_van_giu_override(tmp_path):
    cfg = _cfg_tren_file_tam(tmp_path)
    cfg.set_app_override("llm_api_key", "ollama")

    for _ in range(19):  # đúng số lần load_storytelling_config() trong luồng dựng
        cfg.load_config()

    assert cfg.app["llm_api_key"] == "ollama"


def test_save_config_khong_ghi_override_ra_dia(tmp_path):
    cfg = _cfg_tren_file_tam(tmp_path)
    cfg.set_app_override("llm_api_key", "key-that-cua-nguoi-dung")

    cfg.save_config()

    tren_dia = toml.load(cfg.config_file)
    assert tren_dia["app"]["llm_api_key"] == "", "key chỉ được sống trong bộ nhớ"
    assert cfg.app["llm_api_key"] == "key-that-cua-nguoi-dung", \
        "save_config không được làm mất override trong bộ nhớ"


def test_gia_tri_khong_override_van_theo_file(tmp_path):
    cfg = _cfg_tren_file_tam(tmp_path, app_section={
        "llm_api_key": "", "llm_model": "tu-file"})
    cfg.set_app_override("llm_api_key", "ollama")

    cfg.load_config()

    assert cfg.app["llm_model"] == "tu-file", \
        "override không được đóng băng các khoá khác"


def test_load_storytelling_config_giu_override_tren_singleton():
    """Đúng đường đi thật: singleton `config` + hàm module-level."""
    cfg = config_mod.config
    cu = cfg.app.get("llm_api_key", "")
    overrides_cu = dict(cfg._app_overrides)
    disk_cu = dict(cfg._app_disk_values)
    try:
        cfg.set_app_override("llm_api_key", "ollama")
        config_mod.load_storytelling_config()
        assert cfg.app["llm_api_key"] == "ollama"
    finally:
        cfg._app_overrides = overrides_cu
        cfg._app_disk_values = disk_cu
        cfg.app["llm_api_key"] = cu
