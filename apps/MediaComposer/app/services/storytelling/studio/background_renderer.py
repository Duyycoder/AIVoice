# -*- coding: utf-8 -*-
"""Sinh & cache nền theo location — nền được tái dùng cho mọi cảnh cùng địa điểm.

Logic cache thuần (test đếm số lần render). Việc sinh nền thật (Stable Diffusion)
truyền vào qua `render_fn` để module này không phụ thuộc GPU.
"""
import os
import re
import unicodedata
from typing import Callable, Optional, Tuple

from loguru import logger
from PIL import Image


def safe_location_id(text: str, fallback: str = "loc") -> str:
    """Chuẩn hoá mô tả địa điểm → id an toàn làm tên file cache."""
    s = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    return s[:60] if s else fallback


class BackgroundRenderer:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled

    def cache_path(self, location_id: str) -> str:
        return os.path.join(self.cache_dir, f"{location_id}.png")

    def get_or_render(self,
                      location_id: str,
                      prompt: str,
                      size: Tuple[int, int],
                      render_fn: Callable[[str, Tuple[int, int]], Image.Image]) -> Image.Image:
        """Trả nền từ cache nếu có (và bật cache); ngược lại render + lưu."""
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self.cache_path(location_id) if location_id else ""

        if self.enabled and path and os.path.exists(path):
            try:
                logger.info(f"[Studio][BG] Tái dùng nền cache: {path}")
                return Image.open(path).convert("RGB")
            except Exception as e:
                logger.warning(f"[Studio][BG] Cache lỗi ({e}) — render lại.")

        img = render_fn(prompt, size).convert("RGB")
        if path:
            try:
                img.save(path)
            except Exception as e:
                logger.warning(f"[Studio][BG] Không lưu được nền cache {path}: {e}")
        return img
