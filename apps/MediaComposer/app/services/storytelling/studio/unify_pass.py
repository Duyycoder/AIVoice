# -*- coding: utf-8 -*-
"""Unify pass — img2img denoise thấp trên TOÀN frame sau khi ghép lớp.

Vấn đề nó giải quyết: Studio ghép nhân vật (vẽ riêng, ánh sáng phẳng, nền xám)
lên nền (vẽ riêng, có ánh sáng/tông màu riêng). Dù đã harmonize trung bình màu,
frame vẫn lộ rõ "dán sticker": mép cắt sắc, nhân vật không ăn sáng với nền,
không có bóng đổ, không có hòa sắc khí quyển.

Cách xử lý chuẩn của ngành: chạy lại CẢ frame qua img2img ở strength thấp
(~0.25-0.32). Model giữ nguyên bố cục (vì denoise ít) nhưng vẽ lại bề mặt bằng
MỘT lần thống nhất → ánh sáng, đường nét, tông màu, chất liệu của nhân vật và
nền được kéo về cùng một phong cách; mép matte mờ đi tự nhiên.

Chi phí: img2img chỉ chạy ``int(steps * strength)`` bước thật. Với 16 bước và
strength 0.28 là ~4 bước — rẻ hơn nhiều so với sinh mới một ảnh.

Dùng lại img2img pipeline chia sẻ component của face_detailer nên KHÔNG tốn thêm
VRAM. Mọi lỗi đều trả frame gốc — không bao giờ chặn luồng render.
"""
from typing import Optional

from loguru import logger
from PIL import Image

# Ngưỡng an toàn: strength quá cao sẽ vẽ lại luôn bố cục (mất nhân vật đã ghép).
MAX_SAFE_STRENGTH = 0.45


def clamp_strength(strength: float) -> float:
    """Giữ strength trong vùng bảo toàn bố cục."""
    try:
        value = float(strength)
    except (TypeError, ValueError):
        return 0.0
    if value <= 0:
        return 0.0
    return min(value, MAX_SAFE_STRENGTH)


def build_unify_prompt(style_positive: str, background_prompt: str) -> str:
    """Prompt cho lần hòa trộn: style khóa + bối cảnh, KHÔNG mô tả nhân vật.

    Cố tình không nhắc ngoại hình nhân vật: ở strength thấp model chỉ cần biết
    "vẽ lại bằng phong cách này, trong bối cảnh này". Nhắc nhân vật sẽ khiến nó
    cố sinh thêm người thứ hai ở vùng nền trống.
    """
    from app.services.storytelling.style_lock import strip_style_drift, dedupe_against

    style = ", ".join(t.strip() for t in (style_positive or "").split(",") if t.strip())
    scene = strip_style_drift(background_prompt or "")
    scene = dedupe_against(scene, style)
    # Tag hướng model về "một bức tranh liền mạch" thay vì ảnh ghép.
    cohesion = "coherent lighting, unified color grading, seamless composition"
    return ", ".join(p for p in (style, scene, cohesion) if p)


def unify_frame(pipeline,
                frame: Image.Image,
                *,
                style_positive: str,
                background_prompt: str,
                negative_prompt: str = "",
                strength: float = 0.28,
                num_steps: int = 16,
                guidance_scale: float = 5.0,
                seed: int = -1) -> Image.Image:
    """Chạy img2img strength thấp lên cả frame đã ghép. Lỗi → trả frame gốc."""
    strength = clamp_strength(strength)
    if strength <= 0:
        return frame

    try:
        from app.services.storytelling.face_detailer import _get_img2img
        img2img = _get_img2img(pipeline)
        if img2img is None:
            return frame

        import torch

        prompt = build_unify_prompt(style_positive, background_prompt)
        neg = negative_prompt or "(worst quality:2), (low quality:2), extra person, text, watermark"

        device = getattr(pipeline, "device", "cpu")
        if seed is None or seed < 0:
            seed = int(torch.randint(0, 2147483647, (1,)).item())
        generator = torch.Generator(device=device).manual_seed(seed)

        # IP-Adapter đã gắn vào UNet: phải đưa input trung tính + scale 0 để lần
        # hòa trộn này không kéo khuôn mặt tham chiếu đè lên cả khung hình.
        ip_kwargs = {}
        if getattr(pipeline, "_ip_adapter_loaded", False):
            try:
                pipeline._pipe.set_ip_adapter_scale(0.0)
                ip_kwargs = {"ip_adapter_image": Image.new("RGB", (224, 224), (128, 128, 128))}
            except Exception:
                ip_kwargs = {}

        logger.info(f"[Unify] Hòa trộn frame (strength={strength:.2f}, steps={num_steps}).")
        result = img2img(
            prompt=prompt,
            negative_prompt=neg,
            image=frame.convert("RGB"),
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **ip_kwargs,
        ).images[0]

        if result.size != frame.size:
            result = result.resize(frame.size, Image.LANCZOS)
        return result
    except Exception as e:
        logger.warning(f"[Unify] Bỏ qua hòa trộn frame ({e}) — giữ frame ghép gốc.")
        return frame
    finally:
        # Trả IP-Adapter về mức người dùng cấu hình cho các lớp nhân vật sau.
        try:
            if getattr(pipeline, "_ip_adapter_loaded", False):
                pipeline._pipe.set_ip_adapter_scale(
                    getattr(pipeline, "_ip_adapter_scale", 0.6))
        except Exception:
            pass


def resolve_unify_settings(cfg: dict) -> Optional[dict]:
    """Đọc cấu hình unify pass; trả None nếu tắt."""
    if not bool(cfg.get("studio_unify_pass", True)):
        return None
    strength = clamp_strength(cfg.get("studio_unify_strength", 0.28))
    if strength <= 0:
        return None
    return {
        "strength": strength,
        "num_steps": max(4, int(cfg.get("studio_unify_steps", 16))),
        "guidance_scale": float(cfg.get("studio_unify_guidance", 5.0)),
    }
