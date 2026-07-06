# -*- coding: utf-8 -*-
"""
Face Detailer — tự động phát hiện mặt (anime) và vẽ lại ở độ phân giải cao.

Nguyên lý (giống ADetailer của A1111):
1. Detect mặt bằng lbpcascade_animeface (OpenCV, CPU, ~250KB — tự tải lần đầu).
2. Crop vùng mặt (nới rộng ~65%), phóng lên 512x512.
3. Chạy img2img denoise thấp bằng CHÍNH pipeline SD đang load (chia sẻ UNet/VAE —
   không tốn thêm VRAM), kèm IP-Adapter identity + LoRA nhân vật đang hoạt động.
4. Dán mặt mới về vị trí cũ với mask feather (mép mềm, không lộ vết ghép).

Chạy tự động ngay sau khi sinh ảnh ở Trạm 2 (khi SD còn trên VRAM) — batch không cần
can thiệp. Mọi lỗi ở bất kỳ bước nào → trả ảnh gốc nguyên vẹn (không bao giờ chặn luồng).
"""
import os
# Chống crash native OpenMP kép (torch + OpenCV cùng nạp libiomp5md.dll trên Windows
# → process bị kill thẳng không traceback). Phải set TRƯỚC khi cv2 được dùng.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from loguru import logger

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CASCADE_PATH = os.path.join(_MC_ROOT, "resource", "models", "lbpcascade_animeface.xml")
CASCADE_URLS = [
    "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml",
    "https://cdn.jsdelivr.net/gh/nagadomi/lbpcascade_animeface@master/lbpcascade_animeface.xml",
]

_cascade = None          # cv2.CascadeClassifier cache
_cascade_failed = False  # tránh retry download mỗi ảnh
_warned_once = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned_once:
        _warned_once.add(key)
        logger.warning(msg)


def _download_cascade() -> bool:
    """Tải cascade anime face về resource/models (chạy 1 lần trên mỗi máy)."""
    import urllib.request
    os.makedirs(os.path.dirname(CASCADE_PATH), exist_ok=True)
    for url in CASCADE_URLS:
        try:
            logger.info(f"[FaceDetailer] Đang tải anime face detector: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if b"cascade" not in data[:2000]:
                continue
            with open(CASCADE_PATH, "wb") as f:
                f.write(data)
            logger.info(f"[FaceDetailer] Đã lưu detector: {CASCADE_PATH} ({len(data)//1024} KB)")
            return True
        except Exception as e:
            logger.warning(f"[FaceDetailer] Tải detector thất bại từ {url}: {e}")
    return False


def _ensure_cascade_file() -> bool:
    """Đảm bảo file cascade tồn tại trên đĩa (tự tải nếu thiếu)."""
    global _cascade_failed
    if _cascade_failed:
        return False
    if os.path.exists(CASCADE_PATH) and os.path.getsize(CASCADE_PATH) >= 50_000:
        return True
    # File thiếu hoặc hỏng (tải dở) → tải lại
    if os.path.exists(CASCADE_PATH):
        try:
            os.remove(CASCADE_PATH)
        except Exception:
            pass
    if not _download_cascade():
        _cascade_failed = True
        _warn_once("no_cascade", "[FaceDetailer] Không có anime face detector — bỏ qua vẽ lại mặt. "
                                 "Tải thủ công lbpcascade_animeface.xml vào resource/models/ nếu cần.")
        return False
    return True


def detect_faces(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Trả list bbox (x, y, w, h) các mặt, lớn nhất trước.

    QUAN TRỌNG: detect chạy trong TIẾN TRÌNH CON (scripts/detect_faces_cli.py) —
    OpenCV detectMultiScale + PyTorch chung process trên Windows gây crash native
    (OpenMP kép, kill thẳng không traceback). Cách ly process là fix triệt để:
    tiến trình con có chết cũng chỉ mất bước detect, app không sập.
    """
    if not _ensure_cascade_file():
        return []

    import json
    import subprocess
    import sys
    import tempfile

    logger.info("[FaceDetailer] Bước 2/5: detect mặt (subprocess cách ly OpenCV)...")
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        image.save(tmp_path, format="PNG")

        script = os.path.join(_MC_ROOT, "scripts", "detect_faces_cli.py")
        proc = subprocess.run(
            [sys.executable, script, tmp_path, CASCADE_PATH],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            _warn_once("detect_rc", f"[FaceDetailer] Detect subprocess lỗi (rc={proc.returncode}): "
                                    f"{(proc.stderr or '')[:300]}")
            return []
        boxes = json.loads(proc.stdout.strip() or "[]")
        logger.info(f"[FaceDetailer] Bước 2/5 xong: tìm thấy {len(boxes)} mặt.")
        return [tuple(b) for b in boxes]
    except Exception as e:
        _warn_once("detect_err", f"[FaceDetailer] Detect lỗi: {e} — bỏ qua vẽ lại mặt.")
        return []
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _expand_box(x, y, w, h, img_w, img_h, pad_ratio=0.65):
    """Nới bbox ra mọi phía và ép thành hình vuông (crop vuông → img2img 512 đẹp hơn)."""
    cx, cy = x + w / 2, y + h / 2
    size = max(w, h) * (1 + pad_ratio)
    half = size / 2
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(img_w, cx + half))
    y1 = int(min(img_h, cy + half))
    return x0, y0, x1, y1


def _make_feather_mask(size: Tuple[int, int], feather_px: int) -> Image.Image:
    """Mask ellipse trắng, mép mềm — để dán mặt không lộ viền."""
    from PIL import ImageDraw
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    pad = max(2, feather_px)
    draw.ellipse([pad, pad, size[0] - pad, size[1] - pad], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather_px))


def _clip_similarity(pipeline, img_a: Image.Image, img_b: Image.Image):
    """
    Cosine similarity giữa 2 ảnh qua CLIP image encoder của pipeline (engine clip).
    Trả None nếu encoder không khả dụng — caller tự quyết định bỏ qua kiểm tra.
    """
    try:
        import torch
        pipe = getattr(pipeline, "_pipe", None)
        encoder = getattr(pipe, "image_encoder", None) if pipe else None
        extractor = getattr(pipe, "feature_extractor", None) if pipe else None
        if encoder is None or extractor is None:
            return None
        embs = []
        for img in (img_a, img_b):
            px = extractor(images=img, return_tensors="pt").pixel_values
            px = px.to(device=encoder.device, dtype=encoder.dtype)
            with torch.no_grad():
                emb = encoder(px).image_embeds[0].float().cpu().numpy()
            embs.append(emb)
        a, b = embs
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    except Exception as e:
        logger.debug(f"[FaceDetailer] CLIP similarity không tính được: {e}")
        return None


def _match_color(reference: Image.Image, target: Image.Image) -> Image.Image:
    """Cân bằng mean/std từng kênh màu của target về theo reference (chống lệch tông khi dán)."""
    try:
        ref = np.asarray(reference).astype(np.float32)
        tgt = np.asarray(target).astype(np.float32)
        for c in range(3):
            t_mean, t_std = tgt[..., c].mean(), tgt[..., c].std() + 1e-6
            r_mean, r_std = ref[..., c].mean(), ref[..., c].std() + 1e-6
            tgt[..., c] = (tgt[..., c] - t_mean) / t_std * r_std + r_mean
        return Image.fromarray(np.clip(tgt, 0, 255).astype(np.uint8))
    except Exception:
        return target


def _match_color_mean_only(reference: Image.Image, target: Image.Image) -> Image.Image:
    """Cân màu DỊU: chỉ dịch mean từng kênh, KHÔNG kéo std.

    Match cả std (hàm cũ) khuếch đại tương phản cục bộ → tạo vá màu loang lổ
    trên mặt khi vùng crop chứa cả nền sáng/tối. Dịch mean giữ mối ghép liền màu
    mà không phá nội dung repaint. Chỉ dịch tối đa ±18/kênh để tránh ám màu."""
    try:
        ref = np.asarray(reference).astype(np.float32)
        tgt = np.asarray(target).astype(np.float32)
        for c in range(3):
            delta = ref[..., c].mean() - tgt[..., c].mean()
            delta = max(-18.0, min(18.0, float(delta)))
            tgt[..., c] = tgt[..., c] + delta
        return Image.fromarray(np.clip(tgt, 0, 255).astype(np.uint8))
    except Exception:
        return target


def _build_face_prompt(prompt: str) -> str:
    """Lấy tag ngoại hình từ prompt gốc (bỏ tag bối cảnh/camera) + tag chất lượng mặt."""
    # T3: bỏ cú pháp trọng số (tag:1.3) — img2img ở đây nhận chuỗi thường,
    # giữ nguyên sẽ thành token rác trong CLIP
    from app.services.storytelling.image_generator import StorytellingPipeline
    prompt = StorytellingPipeline._strip_weights(prompt)
    drop_words = ("background", "scenery", "landscape", "wide shot", "long shot",
                  "establishing", "street", "building", "sky", "lighting", "drone",
                  "cinematic", "composition", "raining", "night", "forest", "city")
    tags = [t.strip() for t in prompt.split(",") if t.strip()]
    kept = [t for t in tags if not any(w in t.lower() for w in drop_words)][:14]
    return "masterpiece, best quality, extremely detailed face, sharp clear eyes, " + ", ".join(kept)


def detail_faces(
    pipeline,                      # StorytellingPipeline (đang warmup)
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    face_image: Optional[Image.Image] = None,
    face_embedding: Optional[np.ndarray] = None,
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    strength: float = 0.45,
    max_faces: int = 2,
) -> Image.Image:
    """
    Vẽ lại các mặt trong ảnh. Trả ảnh mới (hoặc ảnh gốc nếu không detect được/bị lỗi).
    Yêu cầu: pipeline._pipe đã warmup (gọi ngay sau generate_draft).
    """
    try:
        if pipeline is None or getattr(pipeline, "_pipe", None) is None:
            return image

        faces = detect_faces(image)
        if not faces:
            return image

        from app.config import load_storytelling_config
        st_config = load_storytelling_config()
        if num_steps is None:
            num_steps = st_config.get("num_inference_steps", 8)
        if guidance_scale is None:
            guidance_scale = st_config.get("guidance_scale", 5.0)

        logger.info("[FaceDetailer] Bước 3/5: tạo img2img pipeline (chia sẻ component)...")
        img2img = _get_img2img(pipeline)
        if img2img is None:
            return image
        logger.info("[FaceDetailer] Bước 3/5 xong.")

        import torch
        img_w, img_h = image.size
        result = image.copy()
        face_prompt = _build_face_prompt(prompt)
        neg = negative_prompt or "(worst quality:2), (low quality:2), blurry, deformed face, bad anatomy"

        # ------------------------------------------------------------------
        # QUAN TRỌNG: bước vẽ mặt LUÔN chạy chế độ Quality riêng.
        # Hyper-SD LoRA chỉ hợp full-denoise từ nhiễu; img2img denoise một phần
        # với Hyper-SD cho ra mặt nát + màu loang lổ. Nên: tắt Hyper-SD tạm thời
        # (giữ FaceID/char LoRA), dùng DPM++ Karras + ~22 steps cho riêng mặt,
        # xong khôi phục nguyên trạng chế độ chính.
        # ------------------------------------------------------------------
        face_steps = int(st_config.get("face_detailer_steps", 14))
        face_cfg = 6.0
        max_faces = int(st_config.get("face_detailer_max_faces", max_faces))
        skip_sim = float(st_config.get("face_detailer_skip_sim", 0.62))
        _switch_to_quality_face_mode(pipeline, img2img)

        try:
            n_done = 0
            done_boxes = []  # các vùng đã vẽ — chống chồng lấn khi có 2+ mặt

            # ------------------------------------------------------------------
            # NÂNG CẤP 2-MẶT: chỉ MẶT GIỐNG NHÂN VẬT NHẤT (CLIP sim với ảnh ref)
            # được áp identity; các mặt khác (poster, nhân vật phụ) vẽ lại GENERIC.
            # Trước đây mọi mặt đều nhận identity → poster bị "nhập vai" nhân vật
            # còn mặt thật dễ hỏng vì conditioning chéo.
            # ------------------------------------------------------------------
            identity_face_idx = 0  # mặc định: mặt lớn nhất
            if face_image is not None and len(faces) > 1:
                best_sim, best_idx = -1.0, 0
                for fi, (fx, fy, fw, fh) in enumerate(faces[:max_faces]):
                    x0, y0, x1, y1 = _expand_box(fx, fy, fw, fh, img_w, img_h)
                    sim = _clip_similarity(pipeline, image.crop((x0, y0, x1, y1)), face_image)
                    if sim is not None and sim > best_sim:
                        best_sim, best_idx = sim, fi
                identity_face_idx = best_idx
                logger.info(f"[FaceDetailer] {len(faces)} mặt — mặt #{identity_face_idx} "
                            f"giống nhân vật nhất (sim={best_sim:.3f}) sẽ nhận identity.")

            # Mặc định CHỈ vẽ lại mặt nhân vật chính — mặt phụ/poster tốn nguyên
            # một lượt img2img (~14s) mà không cần đẹp. Bật lại bằng config:
            # face_detailer_all_faces = true
            only_identity = not st_config.get("face_detailer_all_faces", False)

            for face_i, (fx, fy, fw, fh) in enumerate(faces[:max_faces]):
                if only_identity and len(faces) > 1 and face_i != identity_face_idx:
                    logger.info(f"[FaceDetailer] Bỏ qua mặt #{face_i} (không phải nhân vật chính — "
                                "tiết kiệm ~14s; bật face_detailer_all_faces nếu muốn vẽ mọi mặt).")
                    continue
                # Mặt đã đủ lớn (>45% chiều cao ảnh) → SD gốc đã vẽ tốt, bỏ qua
                if fh > 0.45 * img_h:
                    continue
                # Mặt quá bé (<3% chiều cao) → thường là nhân vật nền, bỏ qua
                if fh < 0.03 * img_h:
                    continue

                x0, y0, x1, y1 = _expand_box(fx, fy, fw, fh, img_w, img_h)

                # Chống chồng lấn: vùng này đè lên mặt đã vẽ → bỏ qua (vẽ đè lên
                # mép feather của mặt trước là nguồn artifact loang lổ)
                overlap = False
                for (dx0, dy0, dx1, dy1) in done_boxes:
                    ix = max(0, min(x1, dx1) - max(x0, dx0))
                    iy = max(0, min(y1, dy1) - max(y0, dy0))
                    if ix * iy > 0.15 * (x1 - x0) * (y1 - y0):
                        overlap = True
                        break
                if overlap:
                    logger.info(f"[FaceDetailer] Bỏ qua mặt #{face_i}: chồng lấn vùng đã vẽ.")
                    continue

                # QUAN TRỌNG: crop từ ẢNH GỐC (image), không phải result — crop từ
                # result sẽ dính mép feather của mặt đã dán trước đó
                crop = image.crop((x0, y0, x1, y1))
                crop_512 = crop.resize((512, 512), Image.LANCZOS)

                # Identity chỉ cho mặt nhân vật; mặt khác vẽ generic (scale 0)
                if face_i == identity_face_idx:
                    identity_kwargs = pipeline._build_identity_kwargs(face_embedding, face_image, face_cfg)
                else:
                    identity_kwargs = pipeline._build_identity_kwargs(None, None, face_cfg)

                # Strength thích ứng: mặt càng nhỏ càng cần vẽ đậm; mặt to chỉ sửa nhẹ
                face_ratio = fh / img_h
                if face_ratio < 0.12:
                    eff_strength = min(0.55, strength + 0.08)
                elif face_ratio > 0.30:
                    eff_strength = max(0.30, strength - 0.05)
                else:
                    eff_strength = strength

                gen = torch.Generator(device=pipeline.device).manual_seed(
                    torch.randint(0, 2147483647, (1,)).item())

                logger.info(f"[FaceDetailer] Bước 4/5: img2img mặt #{face_i} tại ({fx},{fy},{fw}x{fh}) "
                            f"(DPM++ {face_steps} steps, cfg {face_cfg}, strength {eff_strength:.2f}, "
                            f"identity={'CÓ' if face_i == identity_face_idx else 'không'})...")
                out = img2img(
                    prompt=face_prompt,
                    negative_prompt=neg,
                    image=crop_512,
                    strength=eff_strength,
                    num_inference_steps=face_steps,
                    guidance_scale=face_cfg,
                    generator=gen,
                    **identity_kwargs,
                ).images[0]

                # ----------------------------------------------------------
                # CỔNG KIỂM ĐỊNH: mặt vẽ lại hỏng thì VỨT, giữ mặt gốc.
                # Gate 1: output phải detect ra được khuôn mặt (vẽ nát = fail).
                # Gate 2: nếu có ảnh tham chiếu — độ giống nhân vật (CLIP) của
                #         mặt mới phải >= mặt cũ (trừ dung sai nhỏ 0.02).
                # ----------------------------------------------------------
                # ----------------------------------------------------------
                # CỔNG KIỂM ĐỊNH (siết chặt — mặt hỏng thì VỨT, giữ mặt gốc):
                # Gate 1: output phải detect được khuôn mặt.
                # Gate 2: cấu trúc không được lệch quá xa bản gốc
                #         (sim(out, crop) >= 0.55 — bắt các repaint nát/loang lổ).
                # Gate 3: nếu có ref — độ giống nhân vật KHÔNG ĐƯỢC GIẢM
                #         (bỏ dung sai -0.02 cũ: chính nó cho mặt xấu lọt qua).
                # ----------------------------------------------------------
                if not detect_faces(out):
                    logger.warning(f"[FaceDetailer] BỎ mặt #{face_i}: output không detect được "
                                   "khuôn mặt (có thể bị nát) — giữ mặt gốc.")
                    continue

                sim_struct = _clip_similarity(pipeline, out, crop_512)
                if sim_struct is not None and sim_struct < 0.55:
                    logger.warning(f"[FaceDetailer] BỎ mặt #{face_i}: repaint lệch cấu trúc "
                                   f"(sim với bản gốc {sim_struct:.3f} < 0.55) — giữ mặt gốc.")
                    continue

                if face_image is not None and face_i == identity_face_idx:
                    sim_old = _clip_similarity(pipeline, crop_512, face_image)
                    sim_new = _clip_similarity(pipeline, out, face_image)
                    if sim_old is not None and sim_new is not None:
                        if sim_new < sim_old:
                            logger.warning(f"[FaceDetailer] BỎ mặt #{face_i}: kém giống nhân vật hơn "
                                           f"(mới {sim_new:.3f} < cũ {sim_old:.3f}) — giữ mặt gốc.")
                            continue
                        logger.info(f"[FaceDetailer] Kiểm định OK: độ giống {sim_old:.3f} → {sim_new:.3f}")

                logger.info("[FaceDetailer] Bước 5/5: cân màu + dán mặt về ảnh gốc...")
                out_resized = out.resize(crop.size, Image.LANCZOS)
                # Cân màu DỊU: chỉ dịch mean từng kênh (match cả std như trước
                # làm tương phản bị kéo → vá loang lổ trên mặt)
                out_resized = _match_color_mean_only(crop, out_resized)
                feather = max(4, crop.size[0] // 14)
                mask = _make_feather_mask(crop.size, feather)
                result.paste(out_resized, (x0, y0), mask)
                done_boxes.append((x0, y0, x1, y1))
                n_done += 1
        finally:
            # Khôi phục scheduler + LoRA của chế độ chính (Fast/Quality theo tham số gốc)
            _restore_main_mode(pipeline, num_steps, guidance_scale)

        if n_done:
            logger.info(f"[FaceDetailer] Đã vẽ lại {n_done} khuôn mặt.")
        return result

    except Exception as e:
        logger.warning(f"[FaceDetailer] Lỗi ({e}) — dùng ảnh gốc.")
        return image


def _switch_to_quality_face_mode(pipeline, img2img) -> None:
    """Scheduler DPM++ cho img2img + tắt Hyper-SD LoRA (giữ FaceID/char LoRA)."""
    try:
        from diffusers import DPMSolverMultistepScheduler
        sched_cfg = dict(img2img.scheduler.config)
        sched_cfg.pop("algorithm_type", None)
        sched_cfg.pop("final_sigmas_type", None)
        img2img.scheduler = DPMSolverMultistepScheduler.from_config(sched_cfg, use_karras_sigmas=True)
    except Exception as e:
        logger.warning(f"[FaceDetailer] Không đổi được scheduler: {e}")

    try:
        loaded = getattr(pipeline, "_loaded_loras", set())
        if not loaded:
            return
        active, weights = [], []
        if "faceid_lora" in loaded:
            active.append("faceid_lora")
            weights.append(0.65)
        char_slug = getattr(pipeline, "_char_lora_slug", None)
        if char_slug and f"char_{char_slug}" in loaded:
            active.append(f"char_{char_slug}")
            weights.append(0.8)
        if active:
            pipeline._pipe.set_adapters(active, weights)
        else:
            pipeline._pipe.disable_lora()  # chỉ có Hyper-SD → tắt hết cho bước mặt
    except Exception as e:
        logger.warning(f"[FaceDetailer] Không tắt được Hyper-SD LoRA cho bước mặt: {e}")


def _restore_main_mode(pipeline, num_steps, guidance_scale) -> None:
    """Khôi phục scheduler/LoRA của luồng sinh chính sau bước vẽ mặt."""
    try:
        if getattr(pipeline, "_loaded_loras", None):
            try:
                pipeline._pipe.enable_lora()
            except Exception:
                pass
        pipeline._active_mode = None  # buộc _configure_mode chạy lại đầy đủ
        pipeline._configure_mode(num_steps, guidance_scale)
    except Exception as e:
        logger.warning(f"[FaceDetailer] Không khôi phục được chế độ chính: {e}")


def _get_img2img(pipeline):
    """
    Tạo (và cache) StableDiffusionImg2ImgPipeline chia sẻ toàn bộ component với
    pipeline text2img đang load — KHÔNG tốn thêm VRAM, giữ nguyên IP-Adapter + LoRA.
    """
    cached = getattr(pipeline, "_img2img", None)
    if cached is not None:
        return cached
    try:
        from diffusers import AutoPipelineForImage2Image
        img2img = AutoPipelineForImage2Image.from_pipe(pipeline._pipe)
        pipeline._img2img = img2img
        return img2img
    except Exception as e:
        _warn_once("img2img_fail", f"[FaceDetailer] Không tạo được img2img pipeline: {e} — bỏ qua vẽ lại mặt.")
        return None
