# -*- coding: utf-8 -*-
"""Bootstrap + auto-train LoRA cho nhân vật chính — CHẠY 1 LẦN mỗi truyện mới.

Bài toán con-gà-quả-trứng: muốn nhân vật đồng nhất thì cần LoRA, nhưng train LoRA
cần sẵn 10-20 ảnh đồng nhất. Lời giải "tự khởi tạo":

  1. Sinh 1 ảnh SEED (chân dung sạch) từ keywords_en + style, lưu làm ref.
  2. Từ seed ref, dùng IP-Adapter sinh ~12-16 biến thể (đổi góc/pose/biểu cảm/khung)
     — tất cả CÙNG khuôn mặt nhờ IP-Adapter → thành dataset train.
  3. Train LoRA (DreamBooth-style, scripts/train_character_lora.py).

Sau khi có LoRA (`resource/character_loras/<slug>.safetensors`), pipeline TỰ ĐỘNG
dùng nó cho mọi cảnh có nhân vật (image_generator.set_character_lora) ở MỌI chương
của truyện — nên chỉ cần train 1 lần, dùng lại suốt truyện (context gọn hơn).

Idempotent: nếu LoRA đã tồn tại và khớp checkpoint thì bỏ qua (không train lại).
"""
import os
from typing import Callable, List, Optional

from loguru import logger
from PIL import Image

from app.services.storytelling.lora_trainer import LORA_DIR, train_character

# Số ảnh dataset mục tiêu và trần số lần sinh để đạt (một số bị loại bởi cổng chất lượng).
_TARGET_IMAGES = 14
_MAX_ATTEMPTS = 26
# Biến thể khung/góc/biểu cảm để dataset đa dạng (giúp LoRA tổng quát hơn, không kẹt 1 pose).
_VARIATIONS = [
    "portrait, close up face, front view, neutral expression",
    "portrait, close up face, three-quarter view, slight smile",
    "upper body, front view, calm expression",
    "upper body, side profile, looking away",
    "upper body, three-quarter view, serious expression",
    "close up face, looking down, thoughtful",
    "upper body, front view, smiling",
    "full body, standing, front view",
    "full body, standing, three-quarter view",
    "upper body, from slightly above, surprised expression",
    "close up face, front view, determined",
    "upper body, three-quarter view, gentle smile",
    "full body, walking, side view",
    "upper body, front view, talking",
]


def _lora_path(slug: str) -> str:
    return os.path.join(LORA_DIR, f"{slug}.safetensors")


def has_trained_lora(slug: str, checkpoint: str = "") -> bool:
    """True nếu LoRA đã có (và khớp checkpoint nếu có metadata)."""
    path = _lora_path(slug)
    if not os.path.exists(path):
        return False
    if checkpoint:
        import json
        meta_path = os.path.join(LORA_DIR, f"{slug}.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("checkpoint") and meta["checkpoint"] != checkpoint:
                    return False  # LoRA train trên checkpoint khác → coi như chưa có
            except Exception:
                pass
    return True


def _center_square(image: Image.Image) -> Image.Image:
    """Center-crop vuông rồi resize 512 — fallback khi không detect được mặt."""
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 3  # nghiêng lên trên cho hợp chân dung
    return image.crop((left, top, left + side, top + side)).resize((512, 512), Image.LANCZOS)


def _face_crop(image: Image.Image, allow_fallback: bool = True) -> Optional[Image.Image]:
    """Crop mặt vuông 512 nếu tìm được mặt đủ lớn.

    Cascade mặt anime hay miss trên style phẳng; vì mọi ảnh đều đã được IP-Adapter
    ghim theo cùng ref nên khi không thấy mặt ta vẫn giữ ảnh bằng center-crop
    (allow_fallback=True) thay vì bỏ — tránh dataset quá mỏng.
    """
    try:
        from app.services.storytelling.face_detailer import detect_faces, _expand_box
        faces = detect_faces(image)
        if faces and min(faces[0][2], faces[0][3]) >= 96:
            fx, fy, fw, fh = faces[0]
            w, h = image.size
            box = _expand_box(fx, fy, fw, fh, w, h, pad_ratio=0.7)
            return image.crop(box).resize((512, 512), Image.LANCZOS)
    except Exception as e:
        logger.debug(f"[Bootstrap] face crop lỗi: {e}")
    return _center_square(image) if allow_fallback else None


def generate_seed_ref(pipe, ctx_mgr, ctx, slug: str,
                      progress_cb: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Sinh ảnh SEED (chân dung) cho nhân vật rồi lưu làm ref.png. Trả path ref."""
    char = ctx_mgr.get_character(slug)
    if not char:
        return None
    style = ctx.get_positive_prompt()
    neg = ctx.get_negative_prompt()
    appearance = ", ".join(t.strip() for t in char.keywords_en.split(",") if t.strip())
    prompt = (f"{appearance}, solo, 1 person, portrait, close up face, front view, "
              f"neutral expression, simple background, {style}")

    if progress_cb:
        progress_cb(f"Sinh ảnh seed cho '{char.name}'...")
    # Dùng cỡ dọc để mặt to; không truyền IP-Adapter (chưa có ref) → seed thuần prompt.
    img, _ = pipe.generate_draft(prompt=prompt, negative_prompt=neg,
                                 face_embedding=None, face_image=None,
                                 seed=-1, width=512, height=640)
    crop = _face_crop(img) or img
    ref_path = ctx_mgr.set_ref_from_image(slug, crop, source="auto_bootstrap")
    logger.info(f"[Bootstrap] seed ref '{slug}' → {ref_path or '(bị từ chối - đã có ref tay)'}")
    return ref_path or ctx_mgr.get_ref_image_path(slug)


def generate_dataset(pipe, ctx_mgr, ctx, slug: str,
                     target: int = _TARGET_IMAGES,
                     progress_cb: Optional[Callable[[str], None]] = None) -> int:
    """Sinh dataset đa dạng conditioned trên seed ref (IP-Adapter). Trả số ảnh đã lưu."""
    char = ctx_mgr.get_character(slug)
    ref_path = ctx_mgr.get_ref_image_path(slug)
    if not char or not ref_path:
        logger.warning(f"[Bootstrap] '{slug}' thiếu ref — bỏ qua dataset.")
        return 0
    ref_img = Image.open(ref_path).convert("RGB")
    style = ctx.get_positive_prompt()
    neg = ctx.get_negative_prompt()
    appearance = ", ".join(t.strip() for t in char.keywords_en.split(",") if t.strip())

    if hasattr(pipe, "update_ip_adapter_scale"):
        pipe.update_ip_adapter_scale(0.7)  # ưu tiên giữ khuôn mặt cho dataset train

    saved = 0
    for attempt in range(_MAX_ATTEMPTS):
        if saved >= target:
            break
        variation = _VARIATIONS[attempt % len(_VARIATIONS)]
        prompt = (f"{appearance}, solo, 1 person, {variation}, simple background, {style}")
        if progress_cb:
            progress_cb(f"Dataset '{char.name}': {saved}/{target} (thử {attempt+1})")
        try:
            img, _ = pipe.generate_draft(prompt=prompt, negative_prompt=neg,
                                         face_embedding=None, face_image=ref_img,
                                         seed=-1, width=512, height=640)
        except Exception as e:
            logger.warning(f"[Bootstrap] sinh dataset lỗi: {e}")
            continue
        crop = _face_crop(img)
        if crop is None:
            continue
        ctx_mgr.add_dataset_image(slug, crop, source="approved")
        # Ảnh full-body thì lưu thêm bản khung để LoRA học cả dáng người.
        if "full body" in variation:
            ctx_mgr.add_dataset_image(slug, img.resize((512, 512), Image.LANCZOS),
                                      source="approved")
        saved += 1
    logger.info(f"[Bootstrap] dataset '{slug}': {saved} ảnh (dir={ctx_mgr.get_dataset_dir(slug)}).")
    return saved


def bootstrap_and_train(ctx_mgr, slug: str, steps: Optional[int] = None,
                        force: bool = False,
                        progress_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Khởi tạo dataset + train LoRA cho 1 nhân vật. Idempotent (bỏ qua nếu đã có LoRA)."""
    ctx = ctx_mgr._context or ctx_mgr.load_context()
    checkpoint = ctx.checkpoint if ctx else ""
    char = ctx_mgr.get_character(slug)
    if not char:
        logger.warning(f"[Bootstrap] không thấy nhân vật '{slug}'.")
        return False

    if not force and has_trained_lora(slug, checkpoint):
        logger.info(f"[Bootstrap] '{slug}' đã có LoRA — bỏ qua (train 1 lần/truyện).")
        return True

    def _p(msg):
        logger.info(f"[Bootstrap][{slug}] {msg}")
        if progress_cb:
            progress_cb(msg)

    from app.services.storytelling.image_generator import StorytellingPipeline
    pipe = StorytellingPipeline(ctx)
    pipe.warmup()

    # Chỉ tự sinh seed/dataset khi dataset còn mỏng (tôn trọng ảnh user đã có).
    if ctx_mgr.count_dataset_images(slug) < 8:
        if not ctx_mgr.get_ref_image_path(slug):
            generate_seed_ref(pipe, ctx_mgr, ctx, slug, progress_cb=_p)
        generate_dataset(pipe, ctx_mgr, ctx, slug, progress_cb=_p)

    n = ctx_mgr.count_dataset_images(slug)
    if n < 5:
        logger.warning(f"[Bootstrap] '{slug}' chỉ có {n} ảnh (<5) — không train.")
        return False

    _p(f"Train LoRA ({n} ảnh)...")
    # train_character tự release pipeline trước khi train (giải phóng VRAM).
    return train_character(ctx_mgr, slug, steps=steps, progress_cb=progress_cb)


def auto_train_leads(ctx_mgr, lead_slugs: Optional[List[str]] = None,
                     max_leads: int = 2, steps: Optional[int] = None,
                     progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Train LoRA cho các nhân vật chính (mặc định: 2 nhân vật đầu = nam/nữ chính).

    Chạy 1 lần khi tạo truyện mới. Trả {slug: bool}.
    """
    ctx = ctx_mgr._context or ctx_mgr.load_context()
    if lead_slugs is None:
        lead_slugs = [c.slug for c in ctx.characters[:max_leads]]
    results = {}
    for i, slug in enumerate(lead_slugs[:max_leads]):
        if progress_cb:
            progress_cb(f"[{i+1}/{min(len(lead_slugs), max_leads)}] Train nhân vật chính '{slug}'")
        try:
            results[slug] = bootstrap_and_train(ctx_mgr, slug, steps=steps,
                                                progress_cb=progress_cb)
        except Exception as e:
            logger.exception(f"[Bootstrap] auto_train '{slug}' lỗi: {e}")
            results[slug] = False
    return results
