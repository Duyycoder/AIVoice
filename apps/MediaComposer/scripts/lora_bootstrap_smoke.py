# -*- coding: utf-8 -*-
"""Nghiên cứu + kiểm chứng auto-bootstrap LoRA: tạo truyện, train 1 nhân vật chính,
rồi so ảnh CÓ vs KHÔNG LoRA để xác nhận độ đồng nhất/chất lượng tăng.

Chạy: .venv/Scripts/python.exe apps/MediaComposer/scripts/lora_bootstrap_smoke.py [steps]
"""
import os
import shutil
import sys
import time

_MC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = os.path.join(_MC, "storage", "quicktest", "lora")
os.makedirs(OUT, exist_ok=True)
STYLE = os.path.join(_MC, "resource", "image_presets", "storyboard.txt")
SLUG = "lin"


def main():
    from app.services.storytelling.context_manager import ContextManager
    from app.services.storytelling.lora_trainer import LORA_DIR
    from app.services.storytelling import character_bootstrap as cb

    # fresh story
    ctx_mgr = ContextManager("qt_lora")
    if os.path.isdir(ctx_mgr.context_dir):
        shutil.rmtree(ctx_mgr.context_dir)
    ctx = ctx_mgr.create_context("QT Lora", "xianxia")
    shutil.copyfile(STYLE, ctx_mgr.style_file)
    ctx_mgr.add_character(
        name="Lin", description="young swordsman",
        keywords_en="young man, short black hair, black leather jacket, white shirt, "
                    "silver necklace, sharp green eyes, calm expression")
    # remove any stale LoRA from earlier runs so this is a true bootstrap
    for ext in (".safetensors", ".json"):
        p = os.path.join(LORA_DIR, f"{SLUG}{ext}")
        if os.path.exists(p):
            os.remove(p)

    def prog(msg):
        print(f"   · {msg}")

    t0 = time.time()
    ok = cb.bootstrap_and_train(ctx_mgr, SLUG, steps=STEPS, progress_cb=prog)
    dt = time.time() - t0
    print(f">> bootstrap_and_train ok={ok} in {dt:.0f}s ({dt/60:.1f}min), "
          f"dataset={ctx_mgr.count_dataset_images(SLUG)} imgs, steps={STEPS}")
    if not ok:
        return

    # --- verify consistency: 2 scenes WITH LoRA (auto-loaded by set_character_lora) ---
    from app.services.storytelling.image_generator import StorytellingPipeline
    pipe = StorytellingPipeline(ctx)
    pipe.warmup()
    neg = ctx.get_negative_prompt()
    appearance = ctx_mgr.get_character(SLUG).keywords_en
    scenes = [
        ("with_lora_A", "standing in a bamboo forest, full body", SLUG),
        ("with_lora_B", "sitting by a river at sunset, upper body", SLUG),
        ("no_lora_A",   "standing in a bamboo forest, full body", None),
    ]
    for tag, scene_prompt, lora_slug in scenes:
        if hasattr(pipe, "set_character_lora"):
            pipe.set_character_lora(lora_slug)
        prompt = f"{appearance}, solo, 1 person, {scene_prompt}, {ctx.get_positive_prompt()}"
        img, _ = pipe.generate_draft(prompt=prompt, negative_prompt=neg,
                                     face_embedding=None, face_image=None,
                                     seed=1234, width=512, height=640)
        img.save(os.path.join(OUT, f"{tag}.png"))
        print(f"   saved {tag}")
    print(">> out:", OUT)


if __name__ == "__main__":
    main()
