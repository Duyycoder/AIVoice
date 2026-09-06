import os
import sys
import json
import argparse
import gc
import re
import shutil
import unicodedata
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


# Đuôi ngôn ngữ cho tên file .srt — trình phát (VLC/YouTube) nhận diện phụ đề
# theo mã ISO trong tên file: phim.vi.srt, phim.en.srt...
LANG_TAGS = {
    "vietnamese": "vi", "english": "en", "chinese": "zh", "japanese": "ja",
    "korean": "ko", "french": "fr", "spanish": "es", "german": "de",
    "thai": "th", "indonesian": "id", "russian": "ru",
}


def lang_tag(name: str) -> str:
    key = (name or "").strip().lower()
    if key in LANG_TAGS:
        return LANG_TAGS[key]
    slug = unicodedata.normalize("NFKD", key).encode("ascii", "ignore").decode("utf-8")
    slug = re.sub(r"[^a-z0-9]+", "", slug)
    return slug[:8] or "sub"

def main():
    parser = argparse.ArgumentParser(description="CLI Adapter for MediaComposer Autosub & Dubbing Workflows")
    parser.add_argument("--video-path", default="", help="Path to local video file")
    parser.add_argument("--download-url", default="", help="URL of the video to download")
    parser.add_argument("--platform", default="generic", help="Platform platform (bilibili|tiktok|douyin|youtube|generic)")
    parser.add_argument("--output-dir", required=True, help="Output directory to save final result")
    parser.add_argument("--prepare-only", action="store_true", default=False, help="Only download video and extract preview image")
    parser.add_argument("--source-lang", default="English", help="Source video language (English|Chinese)")
    parser.add_argument("--sub-source", default="whisper", choices=["whisper", "ocr", "import"],
                        help="Subtitle generation method (import = dùng file .srt có sẵn)")
    parser.add_argument("--source-srt", default="", help="File .srt nguồn có sẵn (bắt buộc khi --sub-source import)")
    parser.add_argument("--target-lang", default="Vietnamese", help="Ngôn ngữ đích của phụ đề (mặc định Vietnamese)")
    parser.add_argument("--translate-only", action="store_true", default=False,
                        help="Chỉ dịch ra file .srt, không lồng tiếng/không ghi phụ đề vào video")
    parser.add_argument("--no-translate", action="store_true", default=False,
                        help="Ghi thẳng phụ đề nguồn vào video, không gọi LLM dịch (dùng với --sub-source import)")
    parser.add_argument("--srt-out-dir", default="", help="Thư mục chép file .srt nguồn + .srt đã dịch (mặc định = --output-dir)")
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
    
    # Subtitle Customization Styling arguments
    parser.add_argument("--font-name", default=None, help="Subtitle font filename")
    parser.add_argument("--font-size", type=int, default=None, help="Subtitle font size")
    parser.add_argument("--text-color", default=None, help="Subtitle text color (hex or named)")
    parser.add_argument("--stroke-color", default=None, help="Subtitle stroke/border color")
    parser.add_argument("--stroke-width", type=float, default=None, help="Subtitle stroke/border width")
    parser.add_argument("--bg-style", default=None, choices=["None", "Box"], help="Subtitle background style")
    parser.add_argument("--bg-color", default=None, help="Subtitle background box color")
    parser.add_argument("--bg-alpha", type=int, default=None, help="Subtitle background opacity (0-255)")
    parser.add_argument("--sub-position", default=None, choices=["bottom", "top", "center", "custom"], help="Subtitle position on video")
    parser.add_argument("--custom-position", type=float, default=None, help="Custom Y ratio (0-100 from top)")
    parser.add_argument("--cookies-file", default=None, help="Path to cookies file for video downloader")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Use GPU for PaddleOCR")

    args = parser.parse_args()
    
    video_path = args.video_path

    try:
        # 1. Download video if download-url is provided
        if args.download_url:
            from app.services.video_downloader import download_video
            try:
                # Download video to output_dir temporarily
                video_path = download_video(args.download_url, args.output_dir, args.platform, cookies_file=args.cookies_file)
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
        if args.sub_source == "import":
            src_given = os.path.abspath(args.source_srt) if args.source_srt else ""
            if not src_given or not os.path.exists(src_given) or os.path.getsize(src_given) == 0:
                raise RuntimeError(f"Không đọc được file .srt nguồn: {args.source_srt or '(trống)'}")
            # Chép vào task_dir: workflow có thể ghi đè/đọc lại, không đụng file gốc.
            source_srt = os.path.join(task_dir, "source_subtitles.srt")
            shutil.copy(src_given, source_srt)
            log_json("autosub_progress", {
                "message": f"Dùng phụ đề gốc có sẵn: {src_given} — bỏ qua Whisper/OCR.", "percent": 8})
        elif args.sub_source == "ocr":
            ocr_srt_path = os.path.join(task_dir, "ocr_subtitles.srt")
            
            crop_tuple = None
            if args.crop_x >= 0 and args.crop_y >= 0 and args.crop_w > 0 and args.crop_h > 0:
                crop_tuple = (args.crop_x, args.crop_y, args.crop_w, args.crop_h)
                
            ocr_lang = "ch" if args.source_lang.lower() in ["chinese", "zh"] else "en"
            
            # Run OCR in a separate subprocess to avoid CUDA/cuDNN DLL conflicts with PyTorch/Composer
            log_json("autosub_progress", {"message": "Khởi động tiến trình con PaddleOCR...", "percent": 5})
            
            import subprocess
            cmd_ocr = [
                sys.executable,
                "-m", "app.services.subtitle_extractor",
                "--video-path", video_path,
                "--output-srt", ocr_srt_path,
                "--lang", ocr_lang
            ]
            if crop_tuple:
                cmd_ocr.extend([
                    "--crop-x", str(crop_tuple[0]),
                    "--crop-y", str(crop_tuple[1]),
                    "--crop-w", str(crop_tuple[2]),
                    "--crop-h", str(crop_tuple[3])
                ])
            if args.use_gpu:
                cmd_ocr.append("--use-gpu")
                
            # Setup environment with PYTHONPATH containing MediaComposer roots
            env = os.environ.copy()
            mc_root = os.path.dirname(os.path.abspath(__file__))
            paths = [mc_root, os.path.join(mc_root, "app")]
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                paths.append(existing_pythonpath)
            env["PYTHONPATH"] = os.pathsep.join(paths)
            
            proc = subprocess.Popen(
                cmd_ocr,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
                cwd=mc_root
            )
            
            # Pipe output to stdout in real-time for orchestrator tracking
            for line in proc.stdout:
                print(line, end="")
                sys.stdout.flush()
                
            returncode = proc.wait()
                
            if returncode != 0:
                raise RuntimeError(f"OCR Subprocess failed with exit code {returncode}")
                
            source_srt = ocr_srt_path

        # Run translation workflow
        from app.services.composer import composer
        
        log_json("autosub_progress", {"message": "Bắt đầu chạy workflow tạo phụ đề và lồng tiếng...", "percent": 10})
        
        clean_audio_flag = args.clean_audio if args.sub_source == "whisper" else False
        
        workflow_result = composer.run_translation_workflow(
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
            source_srt_override=source_srt,
            font_name=args.font_name,
            font_size=args.font_size,
            text_color=args.text_color,
            stroke_color=args.stroke_color,
            stroke_width=args.stroke_width,
            bg_style=args.bg_style,
            bg_color=args.bg_color,
            bg_alpha=args.bg_alpha,
            position=args.sub_position,
            custom_position=args.custom_position,
            target_lang=args.target_lang,
            translate_only=args.translate_only,
            skip_translate=args.no_translate
        )
        
        # Copy output files to the specified output-dir
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.basename(video_path)
        base_name, _ = os.path.splitext(video_name)

        os.makedirs(args.output_dir, exist_ok=True)
        srt_dir = args.srt_out_dir or args.output_dir
        os.makedirs(srt_dir, exist_ok=True)

        # Phụ đề luôn được xuất ra: người dùng có thể sửa tay rồi ghi lại lần sau,
        # hoặc nạp lên trình phát ngoài mà không cần chạy lại cả workflow.
        viet_srt = os.path.join(task_dir, "vietnamese_subtitles.srt")
        exported = {}
        srt_source_in_task = source_srt or os.path.join(task_dir, "source_subtitles.srt")
        if os.path.exists(srt_source_in_task):
            dst = os.path.join(srt_dir, f"{base_name}.{lang_tag(args.source_lang)}.srt")
            shutil.copy(srt_source_in_task, dst)
            exported["srt_source"] = os.path.abspath(dst)
        if os.path.exists(viet_srt):
            dst = os.path.join(srt_dir, f"{base_name}.{lang_tag(args.target_lang)}.srt")
            shutil.copy(viet_srt, dst)
            exported["srt_translated"] = os.path.abspath(dst)

        # Cảnh báo dịch hỏng (CB2): key/base_url sai thì translate_srt chép nguyên
        # bản gốc mà KHÔNG báo lỗi — video ra đời với phụ đề chưa dịch.
        if os.path.exists(viet_srt) and os.path.exists(srt_source_in_task):
            try:
                with open(viet_srt, "r", encoding="utf-8") as f:
                    viet_lines = f.read().strip()
                with open(srt_source_in_task, "r", encoding="utf-8") as f:
                    src_lines = f.read().strip()
                if viet_lines == src_lines:
                    log_json("autosub_warn", {"message": "Bản dịch trùng bản gốc — kiểm tra API key / model / Base URL dịch!"})
            except OSError:
                pass

        if args.translate_only:
            log_json("autosub_done", {
                "output": exported.get("srt_translated", os.path.abspath(workflow_result)),
                "translate_only": True, **exported})
            sys.exit(0)

        final_video_name = f"{base_name}_autosub_{timestamp}.mp4"
        final_output_path = os.path.join(args.output_dir, final_video_name)
        shutil.copy(workflow_result, final_output_path)

        log_json("autosub_done", {"output": os.path.abspath(final_output_path), **exported})
        sys.exit(0)

    except Exception as e:
        log_json("autosub_error", {"error": str(e)})
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # VRAM release
        if not args.prepare_only:
            try:
                from app.services.subtitle import release_whisper_model
                release_whisper_model()
            except Exception:
                pass
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

if __name__ == "__main__":
    main()
