# -*- coding: utf-8 -*-
"""Khóa style: một nguồn sự thật duy nhất cho art direction của cả truyện.

Trước đây mỗi prompt mang BA tuyên bố style đánh nhau:

1. ``image_generator.generate_draft`` chèn cứng ``"masterpiece, best quality, highres"``
2. ``llm_prompter`` chèn cứng ``"(highly detailed background, cinematic lighting, <model>)"``
3. file style của truyện (vd ``storyboard.txt``) nói ngược lại:
   ``"flat vector illustration, minimal shading, limited color palette"``

Cộng thêm việc LLM được TỰ VIẾT tag style mỗi cảnh, kết quả là mỗi frame ra một
kiểu — đây là nguyên nhân chính của "ảnh không đồng nhất" chứ không phải seed.

Module này chỉ làm hai việc, đều là hàm thuần (test được, không cần GPU):

- ``strip_style_drift``  : bỏ tag style/quality/medium mà LLM tự thêm vào phần nội dung.
- ``build_locked_prompt``: ghép ``[style khóa] + [(hành động:w)] + [nội dung]`` theo
  đúng thứ tự SD1.5 phản hồi tốt nhất (style trước, hành động có trọng số ngay sau).
"""
from typing import Iterable, List

# Tag thuộc về STYLE/CHẤT LƯỢNG/CHẤT LIỆU — chỉ file style của truyện được quyết định.
# LLM viết lại những tag này ở từng cảnh chính là nguồn gây trôi phong cách.
_STYLE_DRIFT_TERMS = frozenset({
    # chất lượng chung
    "masterpiece", "best quality", "high quality", "highres", "high resolution",
    "ultra detailed", "extremely detailed", "highly detailed", "very detailed",
    "detailed", "intricate details", "intricate", "sharp details", "sharp focus",
    "8k", "4k", "uhd", "hdr", "award winning", "beautiful", "stunning",
    "professional", "trending on artstation",
    # chất liệu / medium
    "anime style", "anime", "manga style", "digital art", "digital painting",
    "concept art", "illustration", "oil painting", "watercolor", "watercolour",
    "photorealistic", "photo realistic", "realistic", "hyperrealistic",
    "3d render", "3d", "cgi", "octane render", "unreal engine",
    "cel shading", "flat color", "flat colors", "vector art",
    # ánh sáng / camera thuộc art direction
    "cinematic lighting", "dramatic lighting", "volumetric lighting",
    "studio lighting", "rim lighting", "god rays", "lens flare",
    "depth of field", "bokeh", "film grain", "chromatic aberration",
    "highly detailed background", "detailed background",
})

# Tên model hay bị nhét vào prompt như một tag style (vô nghĩa với CLIP).
_MODEL_NAME_HINTS = ("anything v5", "anythingv5", "dreamshaper", "majicmix",
                     "stable diffusion", "sdxl", "sd1.5")


def _split_tags(prompt: str) -> List[str]:
    return [t.strip() for t in (prompt or "").split(",") if t.strip()]


def _normalize(tag: str) -> str:
    """Bỏ cú pháp trọng số và ký tự thừa để so khớp tag."""
    import re
    s = re.sub(r"\(([^()]+):[0-9.]+\)", r"\1", tag or "")   # (tag:1.3) -> tag
    s = re.sub(r"[()\[\]]", "", s)
    s = re.sub(r":[0-9.]+", "", s)
    return " ".join(s.lower().split())


def strip_style_drift(prompt: str) -> str:
    """Bỏ tag style/quality/medium khỏi phần NỘI DUNG do LLM viết.

    Giữ nguyên mọi tag mô tả chủ thể, bối cảnh, vật thể, thời tiết, hành động —
    chỉ cắt những tag quyết định "vẽ bằng chất liệu gì, đẹp cỡ nào".
    """
    kept = []
    for tag in _split_tags(prompt):
        norm = _normalize(tag)
        if not norm:
            continue
        if norm in _STYLE_DRIFT_TERMS:
            continue
        if any(hint in norm for hint in _MODEL_NAME_HINTS):
            continue
        kept.append(tag)
    return ", ".join(kept)


def dedupe_against(prompt: str, reference: str) -> str:
    """Bỏ tag đã có sẵn trong ``reference`` (thường là style) để khỏi lặp token."""
    ref = {_normalize(t) for t in _split_tags(reference)}
    kept = [t for t in _split_tags(prompt) if _normalize(t) not in ref]
    return ", ".join(kept)


def weight_action(action: str, weight: float = 1.35) -> str:
    """Bọc cụm hành động bằng cú pháp trọng số A1111 để compel nhấn mạnh.

    Tag đã có trọng số sẵn thì giữ nguyên (không bọc chồng).
    """
    tags = _split_tags(action)
    out = []
    for tag in tags:
        if tag.startswith("(") and ":" in tag and tag.endswith(")"):
            out.append(tag)
        else:
            out.append(f"({tag}:{weight:g})")
    return ", ".join(out)


def build_locked_prompt(style_positive: str,
                        content_prompt: str,
                        action: str = "",
                        action_weight: float = 1.35,
                        extra: Iterable[str] = ()) -> str:
    """Ghép prompt cuối theo thứ tự cố định: style → hành động → nội dung.

    SD1.5 ưu tiên token đứng trước, nên style luôn ở đầu (giữ đồng nhất cả truyện)
    và hành động đứng ngay sau với trọng số (đây là thứ trước đây bị chôn ở cuối
    prompt nên model bỏ qua — nhân vật lúc nào cũng chỉ đứng yên).
    """
    style = ", ".join(_split_tags(style_positive))
    content = strip_style_drift(content_prompt)
    content = dedupe_against(content, style)

    parts = [style]
    if action.strip():
        weighted = weight_action(strip_style_drift(action), action_weight)
        parts.append(weighted)
        # Hành động đã lên đầu thì bỏ bản không trọng số còn sót trong nội dung.
        content = dedupe_against(content, strip_style_drift(action))
    parts.append(content)
    parts.extend(str(x).strip() for x in extra if str(x).strip())

    return ", ".join(p for p in parts if p)
