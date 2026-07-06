# -*- coding: utf-8 -*-
"""Thu thập ảnh vào dataset nhân vật cho train LoRA — PHIÊN BẢN T6.

NGUYÊN TẮC CHỐNG NHIỄM DỮ LIỆU:
- Chỉ gọi với ảnh SAU UPSCALE (Trạm 3 / post-process batch), KHÔNG dùng ảnh draft.
- MỌI ảnh (kể cả approved) phải qua cổng chất lượng:
  1. Detect được mặt (anime cascade, subprocess an toàn) với cạnh ngắn ≥ FACE_MIN_PX.
  2. CLIP similarity giữa CROP MẶT và ảnh ref ≥ ngưỡng theo nguồn
     (approved 0.55 — user đã duyệt; auto 0.60 — không có người duyệt).
- Lưu dạng CROP MẶT 512×512 (nguyên liệu lý tưởng cho LoRA train 512px);
  lưu thêm bản full khung khi mặt chiếm >25% chiều cao (ảnh cận cảnh).
- Ảnh user upload tay khi tạo nhân vật KHÔNG đi qua module này (vào thẳng dataset).
"""
import os
import numpy as np
from PIL import Image
from loguru import logger

FACE_MIN_PX = 160          # cạnh ngắn tối thiểu của mặt (trên ảnh sau upscale)
SIM_THRESHOLDS = {"approved": 0.55, "auto": 0.60}
AUTO_MAX_IMAGES = 15       # auto chỉ bổ sung khi dataset còn thiếu
CLOSEUP_FACE_RATIO = 0.25  # mặt >25% chiều cao khung → lưu thêm bản full


def precheck_similarity(ctx_mgr, char_slug: str, image: Image.Image):
    """Tính CLIP similarity (crop mặt vs ref) KHI ENCODER CÒN TRÊN VRAM.

    Dùng cho luồng video/batch: step3 upscale chạy SAU khi SD đã release()
    → encoder không còn. Gọi hàm này ở step2 (sau face detailer), lưu kết quả,
    truyền vào maybe_collect(precomputed_sim=...) ở step3.
    Trả None nếu không detect được mặt / không có ref / encoder chưa load.
    """
    try:
        from app.services.storytelling.face_detailer import detect_faces, _expand_box
        ref_path = ctx_mgr.get_ref_image_path(char_slug)
        if not ref_path:
            return None
        faces = detect_faces(image)
        if not faces:
            return None
        fx, fy, fw, fh = faces[0]
        img_w, img_h = image.size
        x0, y0, x1, y1 = _expand_box(fx, fy, fw, fh, img_w, img_h, pad_ratio=0.6)
        face_crop = image.crop((x0, y0, x1, y1)).resize((512, 512), Image.LANCZOS)
        return _compute_clip_similarity(ctx_mgr, char_slug, face_crop, ref_path)
    except Exception as e:
        logger.debug(f"[DatasetCollector] precheck_similarity lỗi: {e}")
        return None


def maybe_collect(ctx_mgr, char_slug: str, image: Image.Image, source: str,
                  precomputed_sim=None) -> bool:
    """Thu thập ảnh (SAU UPSCALE) vào dataset nhân vật nếu qua cổng chất lượng.

    Args:
        ctx_mgr: ContextManager.
        char_slug: slug nhân vật (caller đảm bảo ảnh chỉ chứa 1 nhân vật).
        image: PIL Image ĐÃ upscale.
        source: "approved" (user đã duyệt ở Trạm 2) | "auto" (batch tự động).

    Returns:
        True nếu ảnh được nhận vào dataset.
    """
    char = ctx_mgr.get_character(char_slug)
    if not char or not getattr(char, "auto_collect", True):
        return False

    if source == "auto" and ctx_mgr.count_dataset_images(char_slug) >= AUTO_MAX_IMAGES:
        logger.debug(f"[DatasetCollector] '{char_slug}' đã đủ {AUTO_MAX_IMAGES} ảnh, bỏ qua auto.")
        return False

    # ------------------------------------------------------------------
    # GATE 1: mặt phải detect được và đủ lớn
    # ------------------------------------------------------------------
    try:
        from app.services.storytelling.face_detailer import detect_faces, _expand_box
    except Exception as e:
        logger.debug(f"[DatasetCollector] Không import được face_detailer: {e}")
        return False

    faces = detect_faces(image)
    if not faces:
        logger.info(f"[DatasetCollector] TỪ CHỐI ({char_slug}/{source}): không detect được mặt.")
        return False

    fx, fy, fw, fh = faces[0]  # mặt lớn nhất
    if min(fw, fh) < FACE_MIN_PX:
        logger.info(f"[DatasetCollector] TỪ CHỐI ({char_slug}/{source}): mặt quá nhỏ "
                    f"({fw}x{fh} < {FACE_MIN_PX}px) — cảnh wide không đủ chuẩn train.")
        return False

    # Crop mặt vuông (bbox × 1.6) → 512×512
    img_w, img_h = image.size
    x0, y0, x1, y1 = _expand_box(fx, fy, fw, fh, img_w, img_h, pad_ratio=0.6)
    face_crop = image.crop((x0, y0, x1, y1)).resize((512, 512), Image.LANCZOS)

    # ------------------------------------------------------------------
    # GATE 2: CLIP similarity giữa CROP MẶT và ảnh ref
    # ------------------------------------------------------------------
    ref_path = ctx_mgr.get_ref_image_path(char_slug)
    threshold = SIM_THRESHOLDS.get(source, 0.60)
    if ref_path:
        # Ưu tiên sim đã đo trước (luồng video: encoder đã bị release lúc này)
        similarity = precomputed_sim
        if similarity is None:
            similarity = _compute_clip_similarity(ctx_mgr, char_slug, face_crop, ref_path)
        if similarity is None:
            # Encoder chưa load → không kiểm được. Approved (đã có mắt người) cho qua,
            # auto (không ai duyệt) thì từ chối cho an toàn.
            if source == "auto":
                logger.info(f"[DatasetCollector] TỪ CHỐI ({char_slug}/auto): "
                            "CLIP encoder chưa load, không kiểm được similarity.")
                return False
        elif similarity < threshold:
            logger.info(f"[DatasetCollector] TỪ CHỐI ({char_slug}/{source}): "
                        f"sim={similarity:.3f} < {threshold} — không đủ giống nhân vật.")
            return False
        else:
            logger.info(f"[DatasetCollector] Gate OK ({char_slug}/{source}): "
                        f"mặt {fw}x{fh}px, sim={similarity:.3f}")
    else:
        logger.debug(f"[DatasetCollector] '{char_slug}' chưa có ref — bỏ qua gate similarity.")

    # ------------------------------------------------------------------
    # Lưu: crop mặt luôn; full khung nếu là ảnh cận cảnh
    # ------------------------------------------------------------------
    path = ctx_mgr.add_dataset_image(char_slug, face_crop, source)
    logger.info(f"[DatasetCollector] NHẬN crop mặt ({char_slug}/{source}): {path}")

    if fh > CLOSEUP_FACE_RATIO * img_h:
        full_path = ctx_mgr.add_dataset_image(char_slug, image, source)
        logger.info(f"[DatasetCollector] NHẬN thêm bản full (mặt {fh/img_h:.0%} khung): {full_path}")

    return True


def _compute_clip_similarity(ctx_mgr, char_slug: str, image: Image.Image,
                             ref_path: str):
    """Cosine similarity giữa ảnh và ref bằng CLIP encoder của pipeline (đã load sẵn).

    Trả None nếu encoder chưa load (không tự load model chỉ để lọc).
    Cache ref embedding vào dataset/.ref_emb.npy — invalidate khi ref mới hơn cache.
    """
    try:
        from app.services.storytelling.image_generator import StorytellingPipeline
        pipe = StorytellingPipeline()
        if not hasattr(pipe, '_pipe') or pipe._pipe is None:
            return None
        encoder = getattr(pipe._pipe, 'image_encoder', None)
        feature_extractor = getattr(pipe._pipe, 'feature_extractor', None)
        if encoder is None or feature_extractor is None:
            return None
        encoder._feature_extractor = feature_extractor
    except Exception:
        return None

    ds_dir = ctx_mgr.get_dataset_dir(char_slug)
    cache_path = os.path.join(ds_dir, ".ref_emb.npy")

    # Invalidate cache nếu ảnh ref mới hơn
    cache_valid = False
    if os.path.exists(cache_path):
        cache_valid = os.path.getmtime(cache_path) >= os.path.getmtime(ref_path)

    if cache_valid:
        ref_emb = np.load(cache_path)
    else:
        ref_img = Image.open(ref_path).convert("RGB")
        ref_emb = _encode_image(encoder, ref_img)
        if ref_emb is None:
            return None
        np.save(cache_path, ref_emb)

    gen_emb = _encode_image(encoder, image)
    if gen_emb is None:
        return None

    return float(
        np.dot(ref_emb.flatten(), gen_emb.flatten()) /
        (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb) + 1e-8)
    )


def _encode_image(encoder, image: Image.Image):
    """Encode ảnh qua CLIP image encoder của pipeline (feature_extractor đi kèm, đúng dtype)."""
    try:
        import torch

        feature_extractor = getattr(encoder, '_feature_extractor', None)
        if feature_extractor is None:
            return None

        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=encoder.device, dtype=encoder.dtype
        )
        with torch.no_grad():
            outputs = encoder(pixel_values=pixel_values)
        return outputs.image_embeds.cpu().float().numpy()
    except Exception as e:
        logger.warning(f"[DatasetCollector] CLIP encode error: {e}")
        return None
