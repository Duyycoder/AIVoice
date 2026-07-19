# -*- coding: utf-8 -*-
"""Sinh nhân vật RIÊNG trên nền phẳng (để chroma-key), khung dọc, mặt to.

Đây là nơi DỒN face-control (IP-Adapter + detailer) — mặt sinh to/isolate nên đẹp
hơn hẳn so với mặt nhỏ trong cảnh rộng. Import Stable Diffusion lazy → chỉ nạp khi
render thật; hàm build_character_prompt là thuần (test được).
"""
import colorsys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Tên màu nền để nhét vào prompt (SD hiểu tên màu tốt hơn mã hex)
BG_COLOR_NAMES = {
    "#00b140": "green",
    "#00ff00": "green",
    "#ff00ff": "magenta",
    "#0000ff": "blue",
}
# Loại bỏ nền phức tạp + hiệu ứng/vải bay lơ lửng (SD anime hay tự thêm khăn/ruy
# băng/hào quang trên nền trống → matte giữ lại thành rác quanh nhân vật).
NEG_ADD = ("complex background, detailed background, scenery, landscape, "
           "multiple people, crowd, extra person, floating ribbon, scarf, "
           "flowing cloth, sash, floating objects, magic effects, glowing particles, "
           "energy aura, sparkles, motion streaks, feathers, "
           "reference sheet, character sheet, multiple views, concept art, "
           "text, english text, japanese text, letters, watermark, signature, "
           "logo, blueprint, grid, annotations, ui, hud, border, frame")


def bg_color_name(hex_color: str) -> str:
    """Mô tả màu hex bằng tên mà text encoder hiểu được.

    Matte vẫn dùng đúng RGB cấu hình; tên gần nhất chỉ giúp model sinh nền có
    cùng họ màu thay vì luôn ép green cho mọi màu tùy chỉnh.
    """
    normalized = (hex_color or "").lower()
    if normalized in BG_COLOR_NAMES:
        return BG_COLOR_NAMES[normalized]
    try:
        value = normalized.lstrip("#")
        if len(value) != 6:
            raise ValueError
        red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    except (TypeError, ValueError):
        return "green"

    hue, saturation, brightness = colorsys.rgb_to_hsv(
        red / 255.0, green / 255.0, blue / 255.0)
    if brightness < 0.15:
        return "black"
    if saturation < 0.15:
        return "white" if brightness > 0.85 else "gray"
    degrees = hue * 360.0
    if degrees < 15 or degrees >= 345:
        return "red"
    if degrees < 45:
        return "orange"
    if degrees < 75:
        return "yellow"
    if degrees < 165:
        return "green"
    if degrees < 195:
        return "cyan"
    if degrees < 255:
        return "blue"
    return "magenta"


_FRAMING_TAGS = {
    "close": "upper body, waist up, portrait framing",
    "medium": "three-quarter body, cowboy shot",
    "full": "full body, standing",
}


def build_character_prompt(appearance: str, color_name: str,
                           framing: str = "full") -> str:
    """Prompt sinh nhân vật theo cỡ cảnh trên nền phẳng đồng nhất."""
    # Giữ prompt ngắn; action/pose được planner đưa lên trước appearance.
    tags = [tag.strip() for tag in (appearance or "").split(",") if tag.strip()]
    appearance = ", ".join(tags[:14])
    framing_tags = _FRAMING_TAGS.get(framing, _FRAMING_TAGS["full"])
    return ", ".join(part for part in (
        appearance, "solo", "1 person", framing_tags, "front view",
        "centered subject", "simple background", f"flat {color_name} background",
        "plain backdrop",
    ) if part)


def build_character_negative_prompt(negative_prompt: str, framing: str) -> str:
    """Bỏ negative tag mâu thuẫn với crop có chủ ý của close/medium shot."""
    tags = [tag.strip() for tag in (negative_prompt or "").split(",") if tag.strip()]
    if framing in {"close", "medium"}:
        conflicts = ("cropped", "out of frame", "cut off")
        tags = [tag for tag in tags
                if not any(term in tag.lower() for term in conflicts)]
    tags.append(NEG_ADD)
    return ", ".join(tags)


class CharacterRenderer:
    def render(self,
               pipeline,
               appearance: str,
               size: Tuple[int, int],
               bg_hex: str,
               *,
               face_image: Optional[Image.Image] = None,
               face_embedding: Optional[np.ndarray] = None,
               negative_prompt: str = "",
               use_detailer: bool = True,
               framing: str = "full") -> Tuple[Image.Image, int]:
        color = bg_color_name(bg_hex)
        prompt = build_character_prompt(appearance, color, framing=framing)
        neg = build_character_negative_prompt(negative_prompt, framing)

        img, seed = pipeline.generate_draft(
            prompt=prompt,
            negative_prompt=neg,
            face_embedding=face_embedding,
            face_image=face_image,
            seed=-1,
            width=size[0],
            height=size[1],
        )

        if use_detailer:
            try:
                from app.services.storytelling.face_detailer import detail_faces
                img = detail_faces(
                    pipeline, img,
                    prompt=prompt, negative_prompt=neg,
                    face_image=face_image, face_embedding=face_embedding,
                )
            except Exception:
                pass
        return img, seed
