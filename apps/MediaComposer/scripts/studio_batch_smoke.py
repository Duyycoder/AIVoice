# -*- coding: utf-8 -*-
"""End-to-end studio run_batch smoke: 4 scenes, 2 locations, 1 character.

Verifies bg-cache reuse (scene 0/1 share a room, 2/3 share a street), per-scene
rembg matte + session reuse, character ref bootstrap, and composite — producing
real video-like frames. Run with the AIVoice venv python from MediaComposer dir.
"""
import os
import shutil
import sys
import time

_MC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC)

_PRESET = sys.argv[1] if len(sys.argv) > 1 else "flat_anime"
OUT = os.path.join(_MC, "storage", "quicktest", f"batch_{_PRESET}")
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT, exist_ok=True)
STYLE = os.path.join(_MC, "resource", "image_presets", f"{_PRESET}.txt")


def main():
    from app.services.storytelling.models import StoryContext, Character, Scene
    from app.services.storytelling.context_manager import ContextManager
    from app.services.storytelling.studio.studio_pipeline import StudioPipeline

    slug = "qt_batch"
    ctx_mgr = ContextManager(slug)
    os.makedirs(ctx_mgr.context_dir, exist_ok=True)
    # clean any prior bg cache / refs so the run is representative
    for sub in ("bg_cache", "characters"):
        p = os.path.join(ctx_mgr.context_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    shutil.copyfile(STYLE, ctx_mgr.style_file)

    char = Character(
        name="Lin", slug="lin", description="young swordsman",
        keywords_en="young man, short black hair, black leather jacket, white shirt, "
                    "necklace, calm expression",
        has_embedding=False)
    ctx = StoryContext(story_name="qt", story_slug=slug, genre="fantasy",
                       checkpoint="stablediffusionapi/anything-v5", characters=[char])
    ctx._style_prompt_path = ctx_mgr.style_file
    ctx_mgr.context = ctx  # some helpers read ctx_mgr.context

    def scene(i, prompt, chars, shot, loc, tod):
        s = Scene(scene_id=i, text_vi="", word_count=10, start_time=0, end_time=5,
                  duration_sec=5, image_prompt=prompt, characters_in_scene=chars,
                  primary_character=chars[0] if chars else "", fallback_level=0,
                  accepted_seed=0, frame_path="", shot_type=shot)
        s._semantic_meta = {"location": loc, "time_of_day": tod}
        return s

    scenes = [
        scene(0, "Lin standing in the dormitory room at night", ["Lin"], "wide",
              "dormitory room, bunk beds, desk lamp, window city lights", "night"),
        scene(1, "Lin close up thinking in the dormitory room at night", ["Lin"], "close",
              "dormitory room, bunk beds, desk lamp, window city lights", "night"),
        scene(2, "empty ancient city street at day", [], "wide",
              "ancient chinese city street, wooden buildings, lanterns", "day"),
        scene(3, "Lin walking on the ancient city street at day", ["Lin"], "medium",
              "ancient chinese city street, wooden buildings, lanterns", "day"),
    ]

    def prog(msg, pct):
        print(f"   [{pct:>3}%] {msg}")

    t0 = time.time()
    StudioPipeline(ctx_mgr, ctx).run_batch(scenes, OUT, progress_cb=prog)
    print(f">> run_batch {time.time()-t0:.1f}s for {len(scenes)} scenes")
    print(">> frames:", sorted(os.listdir(OUT)))
    print(">> bg_cache:", os.listdir(os.path.join(ctx_mgr.context_dir, "bg_cache")))
    print(">> out:", OUT)


if __name__ == "__main__":
    main()
