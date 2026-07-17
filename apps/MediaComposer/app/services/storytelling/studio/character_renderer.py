# -*- coding: utf-8 -*-
"""Sinh nhân vật RIÊNG trên nền phẳng (để chroma-key), khung dọc, mặt to.

Đây là nơi DỒN face-control (IP-Adapter + detailer) — mặt sinh to/isolate nên đẹp
hơn hẳn so với mặt nhỏ trong cảnh rộng. Import Stable Diffusion lazy → chỉ nạp khi
render thật; hàm build_character_prompt là thuần (test được).
"""
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
# Loại bỏ nền phức tạp để chroma-key sạch
NEG_ADD = ("complex background, detailed background, scenery, landscape, "
           "multiple people, crowd, extra person")


def bg_color_name(hex_color: str) -> str:
    return BG_COLOR_NAMES.get((hex_color or "").lower(), "green")


def build_character_prompt(appearance: str, color_name: str) -> str:
    """Prompt sinh nhân vật đứng full-body trên nền phẳng đồng nhất."""
    appearance = (appearance or "").strip().strip(",")
    return (f"masterpiece, best quality, {appearance}, solo, 1 person, full body, "
            f"standing, front view, simple background, flat {color_name} background, "
            f"plain {color_name} backdrop, isolated on {color_name}")


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
               use_detailer: bool = True) -> Tuple[Image.Image, int]:
        color = bg_color_name(bg_hex)
        prompt = build_character_prompt(appearance, color)
        neg = ", ".join(x for x in [negative_prompt, NEG_ADD] if x).strip(", ")

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
