# -*- coding: utf-8 -*-
"""So sánh chất lượng sinh ảnh giữa các checkpoint + số bước, để đẩy chất lượng
studio tiệm cận nanobanana trong giới hạn 6GB VRAM.

Mỗi biến thể sinh 1 character + 1 background (style storyboard), cùng seed, lưu ra
storage/quicktest/quality/<tag>_*.png để so bằng mắt.
"""
import os
import sys
import time

_MC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC)
from PIL import Image  # noqa: E402

OUT = os.path.join(_MC, "storage", "quicktest", "quality")
os.makedirs(OUT, exist_ok=True)
STYLE = os.path.join(_MC, "resource", "image_presets", "storyboard.txt")

APPEARANCE = ("young man, short black hair, black leather jacket, white shirt, "
              "silver necklace, sharp green eyes, calm expression")
BG_PROMPT = ("ancient chinese city street, wooden buildings, red lanterns, "
             "no humans, scenery, empty background")
SEED = 7777

# (tag, checkpoint, steps, guidance)
VARIANTS = [
    ("v5_8step", "stablediffusionapi/anything-v5", 8, 5.0),
    ("v5_22step", "stablediffusionapi/anything-v5", 22, 6.5),
    ("dreamshaper_8step", "dreamshaper-8", 8, 5.0),
    ("dreamshaper_22step", "dreamshaper-8", 22, 6.5),
]


def _ctx(checkpoint):
    from app.services.storytelling.models import StoryContext
    ctx = StoryContext(story_name="qt", story_slug="qt", genre="x",
                       checkpoint=checkpoint)
    ctx._style_prompt_path = STYLE
    return ctx


def run_variant(tag, checkpoint, steps, guidance):
    from app.services.storytelling.image_generator import StorytellingPipeline
    ctx = _ctx(checkpoint)
    pipe = StorytellingPipeline(ctx)   # đổi checkpoint sẽ tự reload
    pipe.warmup(num_steps=steps, guidance_scale=guidance)
    style = ctx.get_positive_prompt()
    neg = ctx.get_negative_prompt()

    t = time.time()
    char_prompt = (f"{APPEARANCE}, solo, 1 person, full body, standing, front view, "
                   f"simple background, flat gray background, {style}")
    char, _ = pipe.generate_draft(prompt=char_prompt, negative_prompt=neg,
                                  face_embedding=None, face_image=None, seed=SEED,
                                  width=512, height=768, num_steps=steps,
                                  guidance_scale=guidance)
    char.save(os.path.join(OUT, f"{tag}_char.png"))
    tc = time.time() - t

    t = time.time()
    bg, _ = pipe.generate_draft(prompt=f"{style}, {BG_PROMPT}", negative_prompt=neg,
                                face_embedding=None, face_image=None, seed=SEED,
                                width=832, height=480, num_steps=steps,
                                guidance_scale=guidance)
    bg.save(os.path.join(OUT, f"{tag}_bg.png"))
    tb = time.time() - t
    print(f">> {tag:20} char={tc:.1f}s bg={tb:.1f}s")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, ckpt, steps, g in VARIANTS:
        if only and only not in tag:
            continue
        try:
            run_variant(tag, ckpt, steps, g)
        except Exception as e:
            print(f">> {tag} FAILED: {e}")
    print(">> out:", OUT)


if __name__ == "__main__":
    main()
