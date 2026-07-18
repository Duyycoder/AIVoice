# -*- coding: utf-8 -*-
"""Test cache nền theo location — thuần, không cần GPU."""
import os
import sys
import tempfile
import shutil

from PIL import Image

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)

from app.services.storytelling.studio.background_renderer import (  # noqa: E402
    BackgroundRenderer, safe_location_id)


def test_safe_location_id():
    assert safe_location_id("Ancient Temple, Night") == "ancient_temple_night"
    assert safe_location_id("Đền cổ") == "en_co"  # bỏ dấu
    assert safe_location_id("") == "loc"          # fallback


class _Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, size):
        self.calls += 1
        return Image.new("RGB", size, (10, 20, 30))


def test_cache_reuse_same_location():
    tmp = tempfile.mkdtemp()
    try:
        r = BackgroundRenderer(tmp, enabled=True)
        fn = _Counter()
        r.get_or_render("loc_a", "p", (32, 32), fn)
        r.get_or_render("loc_a", "p", (32, 32), fn)  # dùng lại cache
        assert fn.calls == 1
        assert os.path.exists(r.cache_path("loc_a"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_different_locations_render_separately():
    tmp = tempfile.mkdtemp()
    try:
        r = BackgroundRenderer(tmp, enabled=True)
        fn = _Counter()
        r.get_or_render("loc_a", "p", (32, 32), fn)
        r.get_or_render("loc_b", "p", (32, 32), fn)
        assert fn.calls == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_disabled_always_renders():
    tmp = tempfile.mkdtemp()
    try:
        r = BackgroundRenderer(tmp, enabled=False)
        fn = _Counter()
        r.get_or_render("loc_a", "p", (32, 32), fn)
        r.get_or_render("loc_a", "p", (32, 32), fn)
        assert fn.calls == 2
        assert not os.path.exists(r.cache_path("loc_a"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
