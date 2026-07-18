# -*- coding: utf-8 -*-
"""Ghép các lớp nhân vật (RGBA) lên nền theo anchor/scale/z-order.

Toán đặt lớp là hàm thuần → test xác định. Hòa màu dùng lại
`_match_color_mean_only` của face_detailer để lớp bớt lệch tông so với nền.
"""
from typing import List, Tuple, Union

from PIL import Image, ImageOps, ImageDraw, ImageFilter

from app.services.storytelling.models import CharacterLayer


def anchor_x_pos(anchor: Union[str, float], frame_w: int, layer_w: int) -> int:
    """Toạ độ x góc trái của lớp theo anchor ngang (đã kẹp trong khung)."""
    if isinstance(anchor, (float, int)):
        x = int(anchor * frame_w - layer_w / 2)
    elif anchor == "left":
        x = 0
    elif anchor == "right":
        x = frame_w - layer_w
    else:  # center
        x = (frame_w - layer_w) // 2
    return max(0, min(x, max(0, frame_w - layer_w)))


def anchor_y_pos(anchor: Union[str, float], frame_h: int, layer_h: int) -> int:
    """Toạ độ y góc trên của lớp theo anchor dọc (đã kẹp trong khung)."""
    if isinstance(anchor, (float, int)):
        y = int(anchor * frame_h - layer_h)
    elif anchor == "middle":
        y = (frame_h - layer_h) // 2
    else:  # bottom (mặc định): đứng đáy khung
        y = frame_h - layer_h
    return max(0, min(y, max(0, frame_h - layer_h)))


def fit_layer_size(frame_h: int, orig_size: Tuple[int, int], scale: float) -> Tuple[int, int]:
    """Kích thước lớp: chiều cao = scale*frame_h (kẹp ≤ frame_h), giữ tỉ lệ."""
    w0, h0 = orig_size
    target_h = max(1, min(frame_h, int(round(scale * frame_h))))
    target_w = max(1, int(round(w0 * target_h / max(1, h0))))
    return target_w, target_h


def _draw_shadow(canvas: Image.Image, x: int, y: int, layer_w: int, layer_h: int, opacity: float):
    """Vẽ bóng ellipse mờ dưới chân nhân vật."""
    if opacity <= 0:
        return
        
    shadow_w = int(layer_w * 0.8)
    shadow_h = int(layer_h * 0.08)
    
    # Canh giữa bóng theo chiều ngang của nhân vật
    sx0 = x + (layer_w - shadow_w) // 2
    sy0 = y + layer_h - shadow_h // 2
    sx1 = sx0 + shadow_w
    sy1 = sy0 + shadow_h
    
    # Tạo mask rời để vẽ ellipse blur
    mask = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([sx0, sy0, sx1, sy1], fill=int(255 * min(1.0, opacity)))
    mask = mask.filter(ImageFilter.GaussianBlur(shadow_h // 2 + 1))
    
    # Composite bóng đen
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 255))
    shadow.putalpha(mask)
    canvas.alpha_composite(shadow)


def composite(background: Image.Image,
              layers: List[Tuple[Image.Image, CharacterLayer]],
              harmonize: bool = True,
              shadow_opacity: float = 0.0) -> Image.Image:
    """Ghép layers (đã tách nền, RGBA) lên background (RGB) theo z-order tăng dần."""
    canvas = background.convert("RGBA")
    fw, fh = canvas.size

    for rgba, layer in sorted(layers, key=lambda t: t[1].z_order):
        c = rgba.convert("RGBA")
        if layer.flip:
            c = ImageOps.mirror(c)

        tw, th = fit_layer_size(fh, c.size, layer.scale)
        if (tw, th) != c.size:
            c = c.resize((tw, th), Image.LANCZOS)
        # Không để lớp rộng hơn khung
        if tw > fw:
            c = c.crop(((tw - fw) // 2, 0, (tw - fw) // 2 + fw, th))
            tw = fw

        if harmonize:
            c = _harmonize(background, c)

        x = anchor_x_pos(layer.anchor_x, fw, tw)
        y = anchor_y_pos(layer.anchor_y, fh, th)
        
        if shadow_opacity > 0:
            _draw_shadow(canvas, x, y, tw, th, shadow_opacity)
            
        canvas.alpha_composite(c, dest=(x, y))

    return canvas.convert("RGB")


def _harmonize(background: Image.Image, layer_rgba: Image.Image) -> Image.Image:
    """Dịch nhẹ tông màu lớp về theo nền (giữ nguyên alpha)."""
    try:
        from app.services.storytelling.face_detailer import _match_color_mean_only
        alpha = layer_rgba.getchannel("A")
        rgb = layer_rgba.convert("RGB")
        rgb = _match_color_mean_only(background.convert("RGB").resize(rgb.size), rgb)
        out = rgb.convert("RGBA")
        out.putalpha(alpha)
        return out
    except Exception:
        return layer_rgba
