# -*- coding: utf-8 -*-
"""Test cổng phát hiện dải chết + điều kiện bật VAE tiling.

Bối cảnh: batch 41 cảnh ngày 24/07 có 9 frame (22%) bị đen mất một dải DỌC rộng
đúng 512 hoặc 384 px trên khung 768 — khớp chính xác biên ô VAE tiling, không
phải lỗi model vẽ.
"""
import os
import sys

import numpy as np
from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.image_generator import (  # noqa: E402
    VAE_TILING_MIN_EDGE, StorytellingPipeline)


def _noise(w=768, h=432, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(40, 210, (h, w, 3), dtype=np.uint8))


def test_clean_image_has_no_dead_region():
    assert StorytellingPipeline.has_dead_region(_noise()) is False


def test_detects_black_tile_band_of_512_on_768_frame():
    """Đúng hình dạng lỗi đã quan sát: scene_019.png."""
    img = _noise()
    arr = np.asarray(img).copy()
    arr[:, :512] = 0
    assert StorytellingPipeline.has_dead_region(Image.fromarray(arr)) is True


def test_detects_384_wide_band():
    img = _noise()
    arr = np.asarray(img).copy()
    arr[:, :384] = 0
    assert StorytellingPipeline.has_dead_region(Image.fromarray(arr)) is True


def test_detects_horizontal_dead_band():
    img = _noise()
    arr = np.asarray(img).copy()
    arr[:200, :] = 0
    assert StorytellingPipeline.has_dead_region(Image.fromarray(arr)) is True


def test_detects_blown_white_band():
    img = _noise()
    arr = np.asarray(img).copy()
    arr[:, :512] = 255
    assert StorytellingPipeline.has_dead_region(Image.fromarray(arr)) is True


def test_dark_ink_artwork_is_not_flagged():
    """Tranh thủy mặc rất tối vẫn hợp lệ — chỉ cột/hàng bão hoà TRỌN VẸN mới tính."""
    arr = np.full((432, 768, 3), 6, dtype=np.uint8)
    # Mảng mực gần đen phủ 2/3 khung nhưng mỗi cột vẫn có vài pixel sáng
    arr[::40, :] = 90
    assert StorytellingPipeline.has_dead_region(Image.fromarray(arr)) is False


def test_thin_dead_band_below_threshold_ignored():
    img = _noise()
    arr = np.asarray(img).copy()
    arr[:, :8] = 0   # 8/768 ~ 1% < ngưỡng 4%
    assert StorytellingPipeline.has_dead_region(Image.fromarray(arr)) is False


class _FakeVae:
    def __init__(self):
        self.tiling = None

    def enable_tiling(self):
        self.tiling = True

    def disable_tiling(self):
        self.tiling = False


class _FakePipe:
    def __init__(self):
        self.vae = _FakeVae()


def _pipeline_with_fake_vae():
    pipe = StorytellingPipeline.__new__(StorytellingPipeline)
    pipe._pipe = _FakePipe()
    return pipe


def test_tiling_off_at_project_render_sizes():
    """768x432 và 512x768 là mọi kích thước pipeline dùng — tiling phải TẮT."""
    pipe = _pipeline_with_fake_vae()
    for w, h in ((768, 432), (512, 768), (704, 528), (576, 704)):
        pipe._set_vae_tiling(w, h)
        assert pipe._pipe.vae.tiling is False, f"{w}x{h} không được bật tiling"


def test_tiling_on_for_genuinely_large_decode():
    pipe = _pipeline_with_fake_vae()
    pipe._set_vae_tiling(VAE_TILING_MIN_EDGE, VAE_TILING_MIN_EDGE)
    assert pipe._pipe.vae.tiling is True


def test_set_tiling_survives_missing_vae():
    pipe = StorytellingPipeline.__new__(StorytellingPipeline)
    pipe._pipe = _FakePipe()
    pipe._pipe.vae = None
    pipe._set_vae_tiling(768, 432)  # không được ném lỗi
