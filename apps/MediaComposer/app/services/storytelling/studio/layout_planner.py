# -*- coding: utf-8 -*-
"""Lập kế hoạch bố cục cảnh (LayerPlan) từ shot_type + danh sách nhân vật.

P0/P1 dùng heuristic thuần (test xác định). Nguồn "llm" (mở rộng schema
llm_prompter) sẽ thêm ở P3; khi LLM thiếu/sai vẫn fallback heuristic này.
"""
from typing import List, Optional

from app.services.storytelling.models import CharacterLayer, LayerPlan

# scale + neo dọc theo cỡ cảnh
_SHOT_DEFAULTS = {
    "close":  (1.0, "middle"),   # cận: nửa người/chân dung ngang tầm mắt
    "medium": (0.8, "bottom"),   # trung: đứng, chiếm ~80% chiều cao
    "wide":   (0.45, "bottom"),  # rộng: nhỏ, đứng trên nền
}


def _distribute_anchor_x(n: int) -> List[str]:
    """Rải nhân vật theo trục ngang để tránh chồng lấn."""
    if n <= 0:
        return []
    if n == 1:
        return ["center"]
    if n == 2:
        return ["left", "right"]
    seq = ["left", "center", "right"]
    return [seq[i % 3] for i in range(n)]


def build_layer_plan(shot_type: str,
                     chars: List[dict],
                     background_prompt: str,
                     location_id: str) -> LayerPlan:
    """chars: list {'slug': str, 'prompt': str} → LayerPlan (heuristic)."""
    base_scale, anchor_y = _SHOT_DEFAULTS.get(
        (shot_type or "wide").lower(), _SHOT_DEFAULTS["wide"])
    anchors = _distribute_anchor_x(len(chars))

    layers: List[CharacterLayer] = []
    for i, ch in enumerate(chars):
        layers.append(CharacterLayer(
            slug=ch.get("slug", ""),
            prompt=ch.get("prompt", ""),
            anchor_x=anchors[i],
            anchor_y=anchor_y,
            scale=base_scale,
            z_order=i,
        ))
    return LayerPlan(
        location_id=location_id,
        background_prompt=background_prompt,
        characters=layers,
        render_mode="studio",
    )


def needs_classic_fallback(n_chars: int,
                           image_prompt: str,
                           max_chars: int,
                           interaction_tags: List[str]) -> Optional[str]:
    """Trả lý do cần render classic cả cảnh, hoặc None nếu ghép lớp được.

    Tiêu chí (quyết định #4):
    - quá nhiều nhân vật (> max_chars) → ghép lớp rối;
    - cảnh chứa tag tương tác vật lý (ôm/đánh/nắm tay...) → lớp rời trông sai.
    """
    if n_chars > max_chars:
        return f"too_many_chars({n_chars}>{max_chars})"
    low = (image_prompt or "").lower()
    for tag in interaction_tags or []:
        if tag and tag.lower() in low:
            return f"interaction_tag:{tag}"
    return None
