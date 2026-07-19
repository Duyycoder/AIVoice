# -*- coding: utf-8 -*-
"""Quick studio-quality probe: render 1 background + 1 character, matte it, composite.

Saves intermediate artifacts so we can eyeball WHERE quality breaks (base gen vs
matting vs composite) and time each stage. Run with the AIVoice venv python from
the MediaComposer dir:

    .venv/Scripts/python.exe apps/MediaComposer/scripts/studio_quicktest.py
"""
import os
import sys
import time

# make `app` importable
_MC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC)

from PIL import Image  # noqa: E402

OUT = os.path.join(_MC, "storage", "quicktest")
os.makedirs(OUT, exist_ok=True)

STYLE = os.path.join(_MC, "resource", "image_presets", "flat_anime.txt")


def _ctx():
    from app.services.storytelling.models import StoryContext, Character
    ctx = StoryContext(
        story_name="qt", story_slug="qt", genre="fantasy",
        checkpoint="stablediffusionapi/anything-v5",
        characters=[Character(
            name="Lin", slug="lin",
            description="young swordsman",
            keywords_en="young man, short black hair, brown leather jacket, "
                        "white shirt, necklace, calm expression",
            has_embedding=False)],
    )
    ctx._style_prompt_path = STYLE
    return ctx


def main():
    from app.config import load_storytelling_config
    from app.services.storytelling.image_generator import StorytellingPipeline
    from app.services.storytelling.studio.character_renderer import CharacterRenderer
    from app.services.storytelling.studio.matting import (
        ChromaMatter, GrabCutMatter, RembgMatter, alpha_coverage)
    from app.services.storytelling.studio.compositor import composite
    from app.services.storytelling.models import CharacterLayer

    cfg = load_storytelling_config()
    ctx = _ctx()
    print(">> style+:", ctx.get_positive_prompt()[:120])

    t0 = time.time()
    pipe = StorytellingPipeline(ctx)
    pipe.warmup()
    print(f">> warmup {time.time()-t0:.1f}s  steps={cfg.get('num_inference_steps')} "
          f"cfg={cfg.get('guidance_scale')}")

    scene_size = (cfg.get("image_width", 768), cfg.get("image_height", 432))
    bg_hex = cfg.get("studio_matte_bg_color", "#00B140")

    # 1) background
    t = time.time()
    bg_prompt = (ctx.get_positive_prompt() +
                 ", dormitory room at night, bunk bed, desk lamp, window city lights, "
                 "no humans, scenery, empty background")
    bg, _ = pipe.generate_draft(prompt=bg_prompt,
                                negative_prompt=ctx.get_negative_prompt(),
                                face_embedding=None, face_image=None, seed=42,
                                width=scene_size[0], height=scene_size[1])
    bg.save(os.path.join(OUT, "01_background.png"))
    print(f">> bg {time.time()-t:.1f}s size={bg.size}")

    # 2) character on flat bg
    t = time.time()
    cr = CharacterRenderer()
    appearance = ("young man, short black hair, brown leather jacket, white shirt, "
                  "necklace, calm expression, " + ctx.get_positive_prompt())
    char, _ = cr.render(pipe, appearance, (512, 768), bg_hex,
                        negative_prompt=ctx.get_negative_prompt(),
                        use_detailer=cfg.get("studio_char_use_detailer", True),
                        framing="full")
    char.save(os.path.join(OUT, "02_character_raw.png"))
    print(f">> char {time.time()-t:.1f}s size={char.size}")

    # 3) chroma matte
    t = time.time()
    cm = ChromaMatter(bg_color=bg_hex,
                      threshold=float(cfg.get("studio_matte_threshold", 0.18)),
                      feather_px=int(cfg.get("studio_matte_feather_px", 3)),
                      despill=bool(cfg.get("studio_matte_despill", True)))
    rgba = cm.cutout(char)
    cov = alpha_coverage(rgba)
    rgba.save(os.path.join(OUT, "03_chroma_rgba.png"))
    rgba.getchannel("A").save(os.path.join(OUT, "03_chroma_alpha.png"))
    print(f">> chroma matte {time.time()-t:.2f}s coverage={cov:.3f}")

    # 3c) rembg (isnet-anime) — engine mới
    t = time.time()
    rm = RembgMatter(model_name="isnet-anime", feather_px=2)
    rgba_rm = rm.cutout(char)
    cov_rm = alpha_coverage(rgba_rm)
    rgba_rm.save(os.path.join(OUT, "03c_rembg_rgba.png"))
    print(f">> rembg matte {time.time()-t:.2f}s coverage={cov_rm:.3f}")

    # 4) composite — dùng rembg matte (engine mới) ở scale wide 0.55
    layer = CharacterLayer(slug="lin", prompt="", anchor_x="center",
                           anchor_y="bottom", scale=0.55, z_order=0, framing="full")
    frame = composite(bg, [(rgba_rm, layer)], harmonize=True)
    frame.save(os.path.join(OUT, "04_composite_rembg.png"))
    print(f">> total {time.time()-t0:.1f}s")
    print(">> out:", OUT)


if __name__ == "__main__":
    main()
