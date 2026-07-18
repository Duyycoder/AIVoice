# -*- coding: utf-8 -*-
"""Lập kế hoạch bố cục cảnh (LayerPlan) từ shot_type + danh sách nhân vật.

P0/P1 dùng heuristic thuần (test xác định). Nguồn "llm" (mở rộng schema
llm_prompter) sẽ thêm ở P3; khi LLM thiếu/sai vẫn fallback heuristic này.
"""
from typing import List, Optional

from app.services.storytelling.models import CharacterLayer, LayerPlan

# scale + neo dọc theo cỡ cảnh
_SHOT_DEFAULTS = {
    "close":  (1.0, "middle", "close"),   # cận: nửa người/chân dung ngang tầm mắt
    "medium": (0.8, "bottom", "medium"),  # trung: 3/4 người, neo đáy
    "wide":   (0.45, "bottom", "full"),   # rộng: toàn thân nhỏ, đứng trên nền
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
    base_scale, anchor_y, framing = _SHOT_DEFAULTS.get(
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
            framing=framing,
        ))
    return LayerPlan(
        location_id=location_id,
        background_prompt=background_prompt,
        characters=layers,
        render_mode="studio",
    )


def build_layer_plan_from_llm(llm_layout: list,
                              chars: List[dict],
                              background_prompt: str,
                              location_id: str,
                              shot_type: str = "wide") -> Optional[LayerPlan]:
    """Parse layout từ LLM. Trả None nếu JSON hỏng/thiếu/sai định dạng."""
    if not isinstance(llm_layout, list) or not llm_layout:
        return None

    import unicodedata
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", s or "").encode(
            "ASCII", "ignore").decode("utf-8").lower().replace(" ", "").replace("_", "")

    char_map = {}
    required_slugs = set()
    for ch in chars:
        slug = ch.get("slug", "")
        if not slug:
            continue
        required_slugs.add(slug)
        # LLM trả display name; slug cũ có thể bị mất ký tự có dấu khi tạo.
        for key in (slug, ch.get("name", "")):
            norm_key = _norm(key)
            if norm_key:
                char_map[norm_key] = ch
    
    _, _, framing = _SHOT_DEFAULTS.get(
        (shot_type or "wide").lower(), _SHOT_DEFAULTS["wide"])
    layers: List[CharacterLayer] = []
    seen_slugs = set()
    for item in llm_layout:
        if not isinstance(item, dict):
            return None
        
        name = item.get("name")
        if not name:
            continue
            
        slug_norm = _norm(name)
        if slug_norm not in char_map:
            continue
            
        try:
            ax = float(item.get("anchor_x", 0.5))
            ay = float(item.get("anchor_y", 0.9))
            scale = float(item.get("scale", 0.8))
            z = int(item.get("z", 0))
            
            if not (0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0 and 0.0 < scale <= 1.0):
                return None
                
            ch_data = char_map[slug_norm]
            slug = ch_data.get("slug", "")
            if slug in seen_slugs:
                return None
            seen_slugs.add(slug)
            pose = str(item.get("prompt") or item.get("pose") or "").strip().strip(",")
            layer_prompt = ch_data.get("prompt", "")
            if pose:
                layer_prompt = ", ".join(x for x in (pose, layer_prompt) if x)
            layers.append(CharacterLayer(
                slug=slug,
                prompt=layer_prompt,
                anchor_x=ax,
                anchor_y=ay,
                scale=scale,
                z_order=z,
                framing=framing,
            ))
        except (ValueError, TypeError):
            return None

    # Layout thiếu dù chỉ một nhân vật cũng phải fallback heuristic; nếu không
    # nhân vật đó sẽ biến mất âm thầm khỏi frame.
    if not layers or seen_slugs != required_slugs:
        return None

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
