# -*- coding: utf-8 -*-
"""LoRA training service — bọc script CLI train_character_lora.py.

Không train tự động khi tạo nhân vật. Chỉ train khi user tick chọn trong pre-flight check.
"""
import os
import sys
import json
import subprocess
import datetime
from loguru import logger

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LORA_DIR = os.path.join(_MC_ROOT, "resource", "character_loras")


def build_instance_prompt(character) -> str:
    """instance_prompt nếu có, không thì clean từ keywords_en (bỏ tag góc chụp, lấy ≤12 tag)."""
    if character.instance_prompt:
        return character.instance_prompt
    tags = [t.strip() for t in character.keywords_en.split(",") if t.strip()]
    blacklist = {
        "upper body", "looking at viewer", "close up", "portrait",
        "headshot", "face focus", "bust shot", "solo focus", "solo",
        "wide shot", "full body", "from behind", "from above",
    }
    cleaned = [t for t in tags if t.lower() not in blacklist][:12]
    return ", ".join(cleaned) if cleaned else "1person"


def get_trainable_characters(ctx_mgr, slugs: list) -> list:
    """Trả danh sách [{slug, name, n_images, eligible, lora_exists, lora_status}]."""
    result = []
    seen = set()
    for slug in slugs:
        if not slug or slug in seen:
            continue
        seen.add(slug)
        char = ctx_mgr.get_character(slug)
        if not char:
            continue
        n_images = ctx_mgr.count_dataset_images(slug)
        lora_path = os.path.join(LORA_DIR, f"{slug}.safetensors")
        result.append({
            "slug": slug,
            "name": char.name,
            "n_images": n_images,
            "eligible": n_images >= 8,
            "lora_exists": os.path.exists(lora_path),
            "lora_status": char.lora_status,
        })
    return result


def train_character(ctx_mgr, slug: str, steps: int = None,
                    progress_cb=None) -> bool:
    """Train LoRA cho 1 nhân vật. Trả True nếu thành công.

    BẮT BUỘC release() pipeline trước khi train (tranh VRAM).
    Train chạy đồng bộ — UI phải cảnh báo user không thao tác browser.
    """
    char = ctx_mgr.get_character(slug)
    if not char:
        logger.error(f"Character '{slug}' not found")
        return False

    n_images = ctx_mgr.count_dataset_images(slug)
    if steps is None:
        if n_images < 10:
            steps = 600
        elif n_images < 20:
            steps = 800
        else:
            steps = 1000

    # 1. Release SD pipeline — BẮT BUỘC giải phóng VRAM
    try:
        from app.services.storytelling.image_generator import StorytellingPipeline
        StorytellingPipeline().release()
    except Exception as e:
        logger.warning(f"Could not release pipeline: {e}")

    # 2. Set status
    char.lora_status = "training"
    ctx_mgr.save_context()

    # 3. Subprocess train
    dataset_dir = ctx_mgr.get_dataset_dir(slug)
    instance_prompt = build_instance_prompt(char)
    context = ctx_mgr._context
    checkpoint = context.checkpoint if context else ""

    script_path = os.path.join(_MC_ROOT, "scripts", "train_character_lora.py")
    cmd = [
        sys.executable, script_path,
        "--character", slug,
        "--images_dir", dataset_dir,
        "--instance_prompt", instance_prompt,
        "--steps", str(steps),
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])

    logger.info(f"[LoRA Train] Starting: {' '.join(cmd)}")
    try:
        # FIX Gap3: stderr=STDOUT để merge vào 1 stream, tránh deadlock buffer
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=_MC_ROOT
        )
        for line in proc.stdout:
            line = line.strip()
            if line and progress_cb:
                progress_cb(line)
            logger.debug(f"[LoRA Train] {line}")
        proc.wait()

        if proc.returncode != 0:
            logger.error(f"[LoRA Train] FAILED (rc={proc.returncode})")
            char.lora_status = "failed"
            ctx_mgr.save_context()
            return False
    except Exception as e:
        logger.error(f"[LoRA Train] Exception: {e}")
        char.lora_status = "failed"
        ctx_mgr.save_context()
        return False

    # 4. Verify output
    lora_path = os.path.join(LORA_DIR, f"{slug}.safetensors")
    if os.path.exists(lora_path):
        char.lora_status = "trained"
        char.lora_trained_at = datetime.datetime.now().isoformat()
        ctx_mgr.save_context()
        # Save metadata cạnh LoRA (để detect checkpoint mismatch)
        meta = {
            "checkpoint": checkpoint,
            "steps": steps,
            "n_images": n_images,
            "trained_at": char.lora_trained_at,
            "instance_prompt": instance_prompt,
        }
        meta_path = os.path.join(LORA_DIR, f"{slug}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"[LoRA Train] SUCCESS: {lora_path}")
        return True
    else:
        char.lora_status = "failed"
        ctx_mgr.save_context()
        return False
