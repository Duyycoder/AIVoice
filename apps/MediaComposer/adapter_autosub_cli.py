import os
import sys
import json
import argparse
import gc
from uuid import uuid4

# Prevent Windows C++ OpenMP abort (OMP: Error #15) when importing torch and cv2 together
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure PYTHONPATH dynamically to import app services correctly
mc_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mc_root)
sys.path.insert(0, os.path.join(mc_root, "app"))

def log_json(event: str, data: dict):
    """Outputs progress log as a JSON string to stdout."""
    print(json.dumps({"event": event, **data}, ensure_ascii=False))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="CLI Adapter for MediaComposer Autosub & Dubbing Workflows")
    parser.add_argument("--video-path", default="", help="Path to local video file")
    parser.add_argument("--download-url", default="", help="URL of the video to download")
    parser.add_argument("--platform", default="generic", help="Platform platform (bilibili|tiktok|douyin|youtube|generic)")
    parser.add_argument("--output-dir", required=True, help="Output directory to save final result")
    parser.add_argument("--prepare-only", action="store_true", default=False, help="Only download video and extract preview image")
    parser.add_argument("--source-lang", default="English", help="Source video language (English|Chinese)")
    parser.add_argument("--sub-source", default="whisper", choices=["whisper", "ocr"], help="Subtitle generation method")
    parser.add_argument("--crop-x", type=int, default=-1, help="Crop region X coord (-1 for full frame)")
    parser.add_argument("--crop-y", type=int, default=-1, help="Crop region Y coord (-1 for full frame)")
    parser.add_argument("--crop-w", type=int, default=-1, help="Crop region Width (-1 for full frame)")
    parser.add_argument("--crop-h", type=int, default=-1, help="Crop region Height (-1 for full frame)")
    parser.add_argument("--burn-method", default="ffmpeg", choices=["ffmpeg", "moviepy"], help="Subtitle burning method")
    parser.add_argument("--clean-audio", action="store_true", default=False, help="Run Demucs vocal isolation (Whisper only)")
    parser.add_argument("--enable-voiceover", action="store_true", default=False, help="Enable translated audio dubbing")
    parser.add_argument("--tts-engine", default="edge", help="TTS Engine (edge|piper|kokoro|vieneu|clone)")
    parser.add_argument("--tts-voice", default="", help="TTS voice name or key")
    parser.add_argument("--auto-clone", action="store_true", default=False, help="Enable auto voice cloning for clone engine")
    parser.add_argument("--ducking-ratio", type=float, default=90.0, help="Audio ducking ratio (0-100)")
    parser.add_argument("--llm-api-key", default="", help="API Key for translation LLM")
    parser.add_argument("--llm-base-url", default="", help="Base URL for translation LLM")
    parser.add_argument("--llm-model", default="", help="Model name for translation LLM")

    args = parser.parse_args()
    
    video_path = args.video_path

    try:
        # 1. Download video if download-url is provided
        if args.download_url:
            from app.services.video_downloader import download_video
            try:
                # Download video to output_dir temporarily
                video_path = download_video(args.download_url, args.output_dir, args.platform)
            except Exception as e:
                log_json("autosub_error", {"error": f"Tải video thất bại: {e}"})
                sys.exit(1)

        if not video_path or not os.path.exists(video_path):
            log_json("autosub_error", {"error": f"Không tìm thấy video tại đường dẫn: {video_path}"})
            sys.exit(1)

        # 2. Prepare only phase (lightweight, no torch/composer imports)
        if args.prepare_only:
            from app.services.subtitle_extractor import grab_preview_frame
            preview_jpg = os.path.join(args.output_dir, "preview.jpg")
            meta = grab_preview_frame(video_path, preview_jpg)
            log_json("prepare_done", {
                "prepared_path": os.path.abspath(video_path),
                "preview_image": os.path.abspath(preview_jpg),
                "width": meta["width"],
                "height": meta["height"],
                "duration": meta["duration"]
            })
            sys.exit(0)

        # 3. Main Workflow execution
        task_id = uuid4().hex[:16]
        
        # Lazy load heavy dependencies
        from app.config import config
        from app.utils import utils
        
        task_dir = utils.task_dir(task_id)

        # CB2: set key config.app["openai_*"] for translate_srt/dubbing (in-memory only)
        if args.llm_api_key:
            config.app["openai_api_key"] = args.llm_api_key
        if args.llm_base_url:
            config.app["openai_base_url"] = args.llm_base_url
        if args.llm_model:
            config.app["openai_model"] = args.llm_model

        source_srt = ""
        if args.sub_source == "ocr":
            from app.services.subtitle_extractor import extract_hardsub_ocr_srt
            ocr_srt_path = os.path.join(task_dir, "ocr_subtitles.srt")
            
            crop_tuple = None
            if args.crop_x >= 0 and args.crop_y >= 0 and args.crop_w > 0 and args.crop_h > 0:
                crop_tuple = (args.crop_x, args.crop_y, args.crop_w, args.crop_h)
                
            ocr_lang = "ch" if args.source_lang.lower() in ["chinese", "zh"] else "en"
            source_srt = extract_hardsub_ocr_srt(
                video_path=video_path,
                output_srt=ocr_srt_path,
                lang=ocr_lang,
                crop=crop_tuple
            )

        # Run translation workflow
        from app.services.composer import MediaComposer
        composer = MediaComposer()
        
        log_json("autosub_progress", {"message": "Bắt đầu chạy workflow tạo phụ đề và lồng tiếng...", "percent": 10})
        
        clean_audio_flag = args.clean_audio if args.sub_source == "whisper" else False
        
        translated_video_path = composer.run_translation_workflow(
            task_id=task_id,
            video_path=video_path,
            source_lang=args.source_lang,
            burn_method=args.burn_method,
            enable_voiceover=args.enable_voiceover,
            tts_engine=args.tts_engine,
            tts_voice=args.tts_voice,
            ducking_ratio=args.ducking_ratio,
            auto_clone=args.auto_clone,
            clean_audio=clean_audio_flag,
            source_srt_override=source_srt
        )
        
        # Copy output file to the specified output-dir
        import shutil
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.basename(video_path)
        base_name, _ = os.path.splitext(video_name)
        final_video_name = f"{base_name}_autosub_{timestamp}.mp4"
        final_output_path = os.path.join(args.output_dir, final_video_name)
        
        shutil.copy(translated_video_path, final_output_path)
        
        # Check translation warning (CB2 warning check)
        viet_srt = os.path.join(task_dir, "vietnamese_subtitles.srt")
        if os.path.exists(viet_srt) and source_srt and os.path.exists(source_srt):
            try:
                with open(viet_srt, "r", encoding="utf-8") as f:
                    viet_lines = f.read().strip()
                with open(source_srt, "r", encoding="utf-8") as f:
                    src_lines = f.read().strip()
                if viet_lines == src_lines:
                    log_json("autosub_warn", {"message": "Bản dịch trùng bản gốc — kiểm tra API key / OpenAI model / Base URL dịch!"})
            except:
                pass
                
        log_json("autosub_done", {"output": os.path.abspath(final_output_path)})
        sys.exit(0)

    except Exception as e:
        log_json("autosub_error", {"error": str(e)})
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # VRAM release
        try:
            from app.services.subtitle import release_whisper_model
            release_whisper_model()
        except:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        gc.collect()

if __name__ == "__main__":
    main()
