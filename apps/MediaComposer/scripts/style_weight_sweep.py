# -*- coding: utf-8 -*-
"""Quét trọng số LoRA phong cách để tìm vùng dùng được.

LoRA phong cách quá mạnh sẽ nuốt cả chủ thể: model chỉ còn vẽ được "chất liệu"
mà không dựng nổi hình người. Script sinh CÙNG một prompt nhân vật và một nền ở
nhiều mức trọng số, cùng seed, để nhìn ra ngưỡng gãy.

    ..\\..\\.venv\\Scripts\\python.exe scripts\\style_weight_sweep.py thuy_mac
"""
import os
import sys
import time

_MC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC)

STYLE_LORA = sys.argv[1] if len(sys.argv) > 1 else "thuy_mac"
PRESET = sys.argv[2] if len(sys.argv) > 2 else "thuy_mac"
WEIGHTS = [0.0, 0.25, 0.4, 0.55, 0.7]
SEED = 7777

OUT = os.path.join(_MC, "storage", "quicktest", f"sweep_{STYLE_LORA}")
STYLE_FILE = os.path.join(_MC, "resource", "image_presets", f"{PRESET}.txt")

APPEARANCE = ("1boy, male focus, young man, black hair, dark robe, "
              "determined expression")
ACTION = "running forward, looking back over shoulder"
BG_PROMPT = ("ancient chinese temple courtyard, stone steps, pine trees, mist, "
             "no humans, scenery")


def main():
    os.makedirs(OUT, exist_ok=True)
    from app.services.storytelling.image_generator import StorytellingPipeline
    from app.services.storytelling.models import StoryContext
    from app.services.storytelling.studio.character_renderer import (
        build_character_negative_prompt, build_character_prompt)

    ctx = StoryContext(story_name="sweep", story_slug="sweep", genre="x",
                       checkpoint="stablediffusionapi/anything-v5")
    ctx._style_prompt_path = STYLE_FILE
    style_pos = ctx.get_positive_prompt()
    style_neg = ctx.get_negative_prompt()
    print(f">> style+: {style_pos[:110]}")

    pipe = StorytellingPipeline(ctx)
    pipe.warmup()

    char_prompt = build_character_prompt(
        f"{APPEARANCE}, {style_pos}", "gray", framing="full", action=ACTION)
    char_neg = build_character_negative_prompt(style_neg, "full")

    for w in WEIGHTS:
        pipe.set_style_lora(STYLE_LORA if w > 0 else None, w)
        pipe.warmup()
        tag = f"w{w:.2f}".replace(".", "")

        t = time.time()
        char, _ = pipe.generate_draft(
            prompt=char_prompt, negative_prompt=char_neg,
            face_embedding=None, face_image=None, seed=SEED,
            width=512, height=768)
        char.save(os.path.join(OUT, f"{tag}_char.png"))

        bg, _ = pipe.generate_draft(
            prompt=f"{style_pos}, {BG_PROMPT}", negative_prompt=style_neg,
            face_embedding=None, face_image=None, seed=SEED,
            width=768, height=432)
        bg.save(os.path.join(OUT, f"{tag}_bg.png"))
        print(f">> weight={w:.2f}  {time.time()-t:.1f}s")

    # Bảng so sánh: mỗi hàng một trọng số
    from PIL import Image, ImageDraw
    cell = 256
    sheet = Image.new("RGB", (cell * len(WEIGHTS), cell + 22), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, w in enumerate(WEIGHTS):
        tag = f"w{w:.2f}".replace(".", "")
        img = Image.open(os.path.join(OUT, f"{tag}_char.png")).resize((cell, cell))
        sheet.paste(img, (i * cell, 22))
        draw.text((i * cell + 6, 5), f"weight {w:.2f}", fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, "_sweep_char.png"))
    print(">> out:", OUT)


if __name__ == "__main__":
    main()
