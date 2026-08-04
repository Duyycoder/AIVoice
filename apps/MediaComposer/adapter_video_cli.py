import os
import sys
import json
import argparse
import gc

# Prevent Windows C++ OpenMP abort (OMP: Error #15) when importing torch and cv2 together
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure PYTHONPATH dynamically to import app services correctly
mc_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mc_root)
sys.path.insert(0, os.path.join(mc_root, "app"))
# Repo root (3 cấp trên MediaComposer) để import được `orchestrator.storage.slugify`
# ngay cả khi chạy adapter độc lập (không qua orchestrator đã set PYTHONPATH).
_repo_root = os.path.abspath(os.path.join(mc_root, "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from app.services.storytelling.context_manager import ContextManager  # noqa: E402
from app.services.storytelling.batch_video_runner import run_batch, scan_batch_dir  # noqa: E402

def log_json(event: str, data: dict):
    """Outputs progress log as a JSON string to stdout."""
    print(json.dumps({"event": event, **data}, ensure_ascii=False))
    sys.stdout.flush()


def count_incomplete_items(report, total_items: int) -> int:
    """Mọi item không có kết quả video_done đều làm adapter exit khác 0."""
    completed = sum(
        1 for item in report.items
        if item.get("status") == "video_done"
        and item.get("video_path")
        and os.path.isfile(item["video_path"])
        and os.path.getsize(item["video_path"]) > 0
    )
    return max(0, int(total_items) - completed)

def main():
    parser = argparse.ArgumentParser(description="Non-interactive CLI Adapter for MediaComposer Video Generation")
    parser.add_argument("--story-name", required=True, help="Name of the story/novel")
    parser.add_argument("--genre", default="tien_hiep", help="Story genre (e.g. tien_hiep, ngon_tinh)")
    parser.add_argument("--input-dir", required=True, help="Input directory containing chapters (.md + .wav)")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated videos")
    parser.add_argument("--bgm-path", default="", help="Path to background music file")
    parser.add_argument("--bgm-volume", type=float, default=0.15, help="Background music volume multiplier")
    parser.add_argument("--enable-upscale", action="store_true", default=True, help="Enable RealESRGAN image upscaling")
    parser.add_argument("--no-upscale", action="store_false", dest="enable_upscale", help="Disable RealESRGAN image upscaling")
    parser.add_argument("--burn-subtitles", action="store_true", default=False, help="Burn subtitles into final video")
    parser.add_argument("--no-subtitles", action="store_false", dest="burn_subtitles", help="Disable subtitles burning")
    parser.add_argument("--use-semantic-split", action="store_true", default=True, help="Use semantic split for scene parsing")
    parser.add_argument("--no-semantic-split", action="store_false", dest="use_semantic_split", help="Disable semantic split")
    parser.add_argument("--extract-characters", action="store_true", default=True, help="Auto extract characters using LLM")
    parser.add_argument("--no-extract-characters", action="store_false", dest="extract_characters", help="Disable auto character extraction")
    parser.add_argument("--enable-face-detailer", action="store_true", default=False, help="Enable Face Detailer")
    parser.add_argument("--no-face-detailer", action="store_false", dest="enable_face_detailer", help="Disable Face Detailer")
    parser.add_argument("--hardware-profile", default="auto", help="Hardware profile: auto, cuda_high, cuda_low, cpu")
    parser.add_argument("--render-mode", default="classic", choices=["classic", "studio"],
                        help="classic = 1 anh/canh (cu); studio = render theo lop (nen + nhan vat) roi ghep")
    parser.add_argument("--style", default="thuy_mac",
                        help="Ten preset trong resource/image_presets (mac dinh: thuy_mac)")
    parser.add_argument("--checkpoint", default="anything-v5", help="Stable Diffusion checkpoint path or hugginface name")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key")
    parser.add_argument("--llm-base-url", default=None, help="LLM base URL")
    parser.add_argument("--llm-model", default=None, help="LLM model name")

    args = parser.parse_args()

    try:
        from app.config import config
        config.storytelling["hardware_profile"] = args.hardware_profile
        config.storytelling["enable_face_detailer"] = args.enable_face_detailer
        config.storytelling["render_mode"] = args.render_mode
        config.save_config()

        # In-memory overrides for LLM (not persisted to config.toml)
        if args.llm_api_key:
            config.app["llm_api_key"] = args.llm_api_key
        if args.llm_base_url:
            config.app["llm_base_url"] = args.llm_base_url
        if args.llm_model:
            config.app["llm_model"] = args.llm_model

        from app.services.storytelling.context_manager import _STORAGE_ENV
        log_json("video_init", {
            "story_name": args.story_name,
            "genre": args.genre,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "storage_env": _STORAGE_ENV
        })

        # Thieu trong so o day KHONG lam pipeline chet: RealESRGAN am tham
        # fallback ve PIL resize (anh mo han) va IP-Adapter mat kha nang giu
        # nhan vat. Vi vay phai tu tai truoc khi render, khong doi setup.bat.
        try:
            from app.services.model_downloader import ensure_models_ready
            status = ensure_models_ready(download_if_missing=True)
            missing = [k for k, v in status.items() if v != "ready"]
            if missing:
                log_json("model_warn", {"missing": missing,
                                        "message": "Mot so model chua tai duoc - chat luong co the giam."})
            else:
                log_json("model_ready", {"count": len(status)})
        except Exception as e:
            log_json("model_warn", {"message": f"Khong kiem tra duoc model: {e}"})

        # Slugify the story name to initialize ContextManager
        from orchestrator.storage import slugify
        story_slug = slugify(args.story_name)

        ctx_mgr = ContextManager(story_slug)
        
        # Auto bootstrapping: Create context if it does not exist (Gap6)
        try:
            context = ctx_mgr.load_context()
            log_json("context_loaded", {"story_slug": story_slug})
        except FileNotFoundError:
            log_json("context_created_start", {"story_slug": story_slug})
            context = ctx_mgr.create_context(args.story_name, args.genre)
            log_json("context_created_success", {"story_slug": story_slug})

        # Update style & checkpoint if specified.
        # apply_style_preset chép preset THẬT vào style_prompt.txt — thiếu bước này
        # thì art_style chỉ là một chuỗi trang trí và mọi truyện đều dùng chung
        # style mặc định ghi lúc tạo context.
        context.art_style = args.style
        context.checkpoint = args.checkpoint
        if ctx_mgr.apply_style_preset(args.style):
            log_json("style_applied", {"style": args.style})
        ctx_mgr.save_context(context)

        # Scan input directory for batch items (md + audio)
        items = scan_batch_dir(args.input_dir)
        if not items:
            log_json("video_warn", {"message": f"No valid batch items (md + audio pairs) found in {args.input_dir}"})
            sys.exit(0)

        # Extract characters if none exist and enabled
        if args.extract_characters and not context.characters:
            log_json("video_progress", {"message": "Đang phân tích kịch bản để tự động nhận diện nhân vật (LLM)...", "percent": 0})
            from app.services.storytelling.character_extractor import process_chapters_and_extract_characters
            
            # Read first 5 chapters to avoid context limits
            chapter_texts = []
            for item in items[:5]:
                md_path = os.path.join(args.input_dir, item.md_path)
                with open(md_path, "r", encoding="utf-8") as f:
                    chapter_texts.append(f.read())
                    
            try:
                extracted_chars = process_chapters_and_extract_characters(
                    chapter_texts=chapter_texts,
                    story_name=args.story_name,
                    genre=args.genre,
                    enable_web_search=True
                )
                if extracted_chars:
                    from app.services.storytelling.models import Character
                    from orchestrator.storage import slugify
                    for c_dict in extracted_chars:
                        name = c_dict.get("name", "")
                        if not name:
                            continue
                        char_obj = Character(
                            name=name,
                            slug=slugify(name),
                            keywords_en=c_dict.get("keywords_en", ""),
                            description=c_dict.get("description", ""),
                            has_embedding=False
                        )
                        context.characters.append(char_obj)
                    ctx_mgr.save_context(context)
                    log_json("video_progress", {"message": f"Đã nhận diện thành công {len(context.characters)} nhân vật.", "percent": 0})
                else:
                    log_json("video_warn", {"message": "Không nhận diện được nhân vật nào từ kịch bản."})
            except Exception as e:
                log_json("video_warn", {"message": f"Lỗi khi nhận diện nhân vật: {e}"})

        log_json("video_batch_start", {
            "total_items": len(items),
            "stems": [it.stem for it in items]
        })

        options = {
            "bgm_path": args.bgm_path,
            "bgm_volume": args.bgm_volume,
            "enable_upscale": args.enable_upscale,
            "burn_subtitles": args.burn_subtitles,
            "use_semantic_split": args.use_semantic_split,
            "use_whisper": False  # Enforced: Do not use Whisper globally (M1 policy)
        }

        # Run Batch Runner with a custom progress callback to print JSON logs
        def progress_callback(msg: str, percent: int):
            log_json("video_progress", {"message": msg, "percent": percent})

        report = run_batch(
            ctx_mgr=ctx_mgr,
            items=items,
            output_dir=args.output_dir,
            options=options,
            progress_cb=progress_callback
        )

        # Output final execution details
        failed_count = count_incomplete_items(report, len(items))
        log_json("video_batch_completed", {
            "total": len(report.items),
            "failed": failed_count,
            "status": "success" if failed_count == 0 else "partial_failure"
        })
        if failed_count:
            raise SystemExit(1)

    except Exception as e:
        log_json("video_error", {"error": str(e)})
        sys.exit(1)
    finally:
        # Crucial VRAM and RAM cleanup to release hardware resources after execution (Gap5)
        try:
            from app.services.storytelling.image_generator import StorytellingPipeline
            StorytellingPipeline().release()
        except Exception:
            pass
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        log_json("hardware_released", {"status": "success"})

if __name__ == "__main__":
    main()
