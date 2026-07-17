# -*- coding: utf-8 -*-
"""Tách nền cho lớp nhân vật — chroma-key trên nền phẳng.

Nhân vật được sinh trên một nền màu đồng nhất (vd xanh #00B140); module này key
theo khoảng cách màu tới màu nền → alpha, kèm despill (khử ám màu) + feather (mềm mép).

Thuần numpy/PIL — KHÔNG cần GPU. `Matter` là interface để sau này thay bằng model
tách nền tốt hơn (isnet-anime, InSPyReNet...) mà không phải sửa compositor.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

# Chuẩn hoá: khoảng cách RGB tối đa = sqrt(3 * 255^2)
_MAX_RGB_DIST = float(np.sqrt(3.0) * 255.0)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """'#00B140' → (0, 177, 64). Chấp nhận có/không '#'."""
    s = (hex_color or "").lstrip("#").strip()
    if len(s) != 6:
        return (0, 177, 64)  # mặc định xanh
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (0, 177, 64)


class Matter(ABC):
    """Interface tách nền: RGB → RGBA (kênh alpha là mặt nạ nhân vật)."""

    @abstractmethod
    def cutout(self, image: Image.Image) -> Image.Image:
        ...


class ChromaMatter(Matter):
    def __init__(self, bg_color: str = "#00B140", threshold: float = 0.18,
                 feather_px: int = 3, despill: bool = True, ramp: float = 0.10):
        self.bg_rgb = np.asarray(hex_to_rgb(bg_color), dtype=np.float32)
        self.threshold = float(threshold)
        self.feather_px = int(feather_px)
        self.despill = bool(despill)
        self.ramp = max(1e-3, float(ramp))  # độ rộng dốc chuyển alpha

    def cutout(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

        # Khoảng cách màu tới nền, chuẩn hoá 0..1
        dist = np.linalg.norm(rgb - self.bg_rgb, axis=2) / _MAX_RGB_DIST
        # Gần nền (dist < threshold) → alpha 0; xa nền → alpha 1, dốc mềm theo `ramp`
        alpha = np.clip((dist - self.threshold) / self.ramp, 0.0, 1.0)

        if self.despill:
            rgb = self._despill(rgb)

        alpha_img = Image.fromarray((alpha * 255.0).astype(np.uint8))  # 2D uint8 → "L"
        if self.feather_px > 0:
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(self.feather_px))

        out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")  # 3D → "RGB"
        out.putalpha(alpha_img)
        return out

    def _despill(self, rgb: np.ndarray) -> np.ndarray:
        """Khử ám màu nền: giới hạn kênh trội của nền ≤ max của 2 kênh còn lại."""
        ch = int(np.argmax(self.bg_rgb))
        others = [i for i in range(3) if i != ch]
        cap = np.maximum(rgb[..., others[0]], rgb[..., others[1]])
        rgb = rgb.copy()
        rgb[..., ch] = np.minimum(rgb[..., ch], cap)
        return rgb


def alpha_coverage(rgba: Image.Image, opaque_thresh: int = 16) -> float:
    """Tỉ lệ pixel 'giữ lại' (alpha > ngưỡng) — dùng cho cổng kiểm định key hỏng."""
    if rgba.mode != "RGBA":
        return 1.0
    a = np.asarray(rgba.getchannel("A"))
    return float((a > opaque_thresh).mean())
