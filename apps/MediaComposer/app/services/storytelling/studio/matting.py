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
            rgb = self._despill(rgb, alpha)

        alpha_img = Image.fromarray((alpha * 255.0).astype(np.uint8))  # 2D uint8 → "L"
        if self.feather_px > 0:
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(self.feather_px))

        out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")  # 3D → "RGB"
        out.putalpha(alpha_img)
        return out

    def _despill(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Khử màu nền ở mép bán trong suốt, giữ màu hợp lệ trong foreground."""
        # Màu chroma có thể có nhiều kênh trội (magenta = đỏ + lam), vì vậy
        # không được chỉ xử lý argmax đầu tiên. Kênh trội là kênh cao hơn kênh
        # thấp nhất của màu nền một khoảng đủ lớn; nền gần xám thì bỏ despill.
        dominant = self.bg_rgb > (float(self.bg_rgb.min()) + 64.0)
        others = np.flatnonzero(~dominant)
        if not dominant.any() or not len(others):
            return rgb
        cap = np.max(rgb[..., others], axis=-1)
        rgb = rgb.copy()
        edge_strength = 1.0 - alpha
        for channel in np.flatnonzero(dominant):
            reduced = np.minimum(rgb[..., channel], cap)
            rgb[..., channel] = (
                rgb[..., channel] * alpha + reduced * edge_strength)
        return rgb


class GrabCutMatter(Matter):
    """Fallback CPU nhanh khi model tạo nền đơn giản nhưng không đúng màu chroma."""

    def __init__(self, iterations: int = 5, feather_px: int = 3,
                 max_analysis_size: int = 384):
        self.iterations = max(1, int(iterations))
        self.feather_px = max(0, int(feather_px))
        self.max_analysis_size = max(64, int(max_analysis_size))

    def cutout(self, image: Image.Image) -> Image.Image:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("GrabCut fallback cần opencv-python") from e

        rgb_full = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb_full.shape[:2]
        if h < 4 or w < 4:
            return image.convert("RGBA")

        # Ảnh phẳng hoàn toàn (thường là chroma hỏng trong test/generation) không
        # có foreground để GrabCut học; trả mask rỗng ngay thay vì chạy rất lâu.
        if float(rgb_full.reshape(-1, 3).std(axis=0).max()) < 2.0:
            out = image.convert("RGBA")
            out.putalpha(Image.new("L", image.size, 0))
            return out

        scale = min(1.0, self.max_analysis_size / float(max(h, w)))
        if scale < 1.0:
            sw, sh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
            rgb = cv2.resize(rgb_full, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            rgb = rgb_full
        ah, aw = rgb.shape[:2]

        # Viền 1px chắc chắn là nền; phần trong là vùng GrabCut tự phân loại.
        mask = np.full((ah, aw), cv2.GC_PR_BGD, dtype=np.uint8)
        rect = (1, 1, aw - 2, ah - 2)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(rgb, mask, rect, bg_model, fg_model,
                    self.iterations, cv2.GC_INIT_WITH_RECT)
        alpha = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Bỏ các đốm nền nhỏ nhưng giữ chi tiết tóc/áo đã nối với chủ thể.
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        if (aw, ah) != (w, h):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha_img = Image.fromarray(alpha)
        if self.feather_px:
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(self.feather_px))

        out = image.convert("RGBA")
        out.putalpha(alpha_img)
        return out


def alpha_coverage(rgba: Image.Image, opaque_thresh: int = 16) -> float:
    """Tỉ lệ pixel 'giữ lại' (alpha > ngưỡng) — dùng cho cổng kiểm định key hỏng."""
    if rgba.mode != "RGBA":
        return 1.0
    a = np.asarray(rgba.getchannel("A"))
    return float((a > opaque_thresh).mean())
