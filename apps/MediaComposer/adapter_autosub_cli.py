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

def _doc_srt_don_gian(path):
    """[{t_vao, t_ra, text}] từ file .srt (đủ cho SRT do editor ghi ra)."""
    def giay(s):
        h, m, rest = s.strip().replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    ds = []
    with open(path, encoding="utf-8-sig") as fh:
        khoi = [k for k in fh.read().replace("\r\n", "\n").split("\n\n") if k.strip()]
    for k in khoi:
        dong = [d for d in k.split("\n") if d.strip()]
        moc = next((i for i, d in enumerate(dong) if "-->" in d), None)
        if moc is None:
            continue
        a, b = dong[moc].split("-->")
        ds.append({"t_vao": giay(a), "t_ra": giay(b.split()[0]), "text": " ".join(dong[moc + 1:]).strip()})
    return ds


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
    parser.add_argument("--whisper-model", default="", help="Whisper model name or path (e.g. base, medium, qbsmlabs/PhoWhisper-small)")
    parser.add_argument("--whisper-device", default="", choices=["", "auto", "cuda", "cpu"], help="Device for Whisper (auto/cuda/cpu)")
    parser.add_argument("--enable-voiceover", action="store_true", default=False, help="Enable translated audio dubbing")
    parser.add_argument("--voiceover-only", action="store_true", default=False,
                        help="Editor: chỉ tạo file GIỌNG lồng tiếng (.wav) rồi chép ra --audio-out-dir, không ghi vào video")
    parser.add_argument("--doc-van-ban-only", action="store_true", default=False,
                        help="Chỉ đọc văn bản từ file (không cần video)")
    parser.add_argument("--minh-hoa-only", action="store_true", default=False, help="Chỉ tải video minh hoạ")
    parser.add_argument("--minh-hoa-nguon", default="pexels", choices=["pexels", "pixabay", "coverr"], help="Nguồn video minh hoạ")
    parser.add_argument("--minh-hoa-ti-le", default="9:16", choices=["9:16", "16:9", "1:1"], help="Tỉ lệ video minh hoạ")
    parser.add_argument("--pexels-api-key", default="", help="Pexels API Key")
    parser.add_argument("--pixabay-api-key", default="", help="Pixabay API Key")
    parser.add_argument("--coverr-api-key", default="", help="Coverr API Key")
    parser.add_argument("--van-ban-file", default="", help="File txt chứa văn bản cần đọc")
    parser.add_argument("--audio-out-dir", default="", help="Thư mục chép file giọng lồng tiếng (mặc định = --output-dir)")
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
    parser.add_argument("--tach-giong-only", action="store_true", default=False, help="Chỉ tách giọng thành giong/nhac.wav")
    parser.add_argument("--lam-net-only", action="store_true", default=False, help="Chỉ làm nét bằng RealESRGAN")
    parser.add_argument("--lam-net-kieu", default="nhanh", choices=["nhanh", "ai", "ai_video"], help="Thuật toán làm nét")
    parser.add_argument("--lam-net-do-phan-giai", default="Gốc", choices=["Gốc", "720p (HD)", "1080p (Full HD)", "1440p (2K)", "2160p (4K)"], help="Độ phân giải đầu ra")
    parser.add_argument("--can-gio-only", action="store_true", default=False, help="Chỉ lấy word timestamps để căn giờ")
    parser.add_argument("--tao-anh-only", action="store_true", default=False, help="Chỉ tạo ảnh AI")
    parser.add_argument("--anh-prompt", default="", help="Prompt tạo ảnh")
    parser.add_argument("--anh-negative", default="", help="Negative prompt")
    parser.add_argument("--anh-model", default="", help="Model tạo ảnh")
    parser.add_argument("--anh-rong", type=int, default=1024, help="Chiều rộng ảnh")
    parser.add_argument("--anh-cao", type=int, default=1024, help="Chiều cao ảnh")
    parser.add_argument("--anh-so", type=int, default=1, help="Số ảnh")
    parser.add_argument("--anh-steps", type=int, default=20, help="Số steps")
    parser.add_argument("--anh-guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--anh-seed", type=int, default=-1, help="Seed")

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

        if not (args.tao_anh_only or args.doc_van_ban_only or args.minh_hoa_only):   # các việc này không có video nguồn
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

        if getattr(args, "doc_van_ban_only", False):
            van_ban = ""
            if args.van_ban_file and os.path.exists(args.van_ban_file):
                with open(args.van_ban_file, "r", encoding="utf-8") as f:
                    van_ban = f.read().strip()
            if not van_ban:
                raise RuntimeError("File văn bản trống hoặc không tìm thấy.")
                
            audio_dir = args.audio_out_dir or args.output_dir
            os.makedirs(audio_dir, exist_ok=True)
            import time
            stamp = int(time.time())
            out_wav = os.path.join(audio_dir, f"doc_{stamp}.wav")
            
            engine_name = args.tts_engine or "edge"
            voice_name = args.tts_voice or ""
            
            # Setup path for src.engines
            project_root = os.path.dirname(os.path.dirname(mc_root))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                
            if 'src' in sys.modules:
                src_mod = sys.modules['src']
                if getattr(src_mod, '__file__', None) is None or 'site-packages' in str(src_mod.__file__):
                    del sys.modules['src']
                    for k in list(sys.modules.keys()):
                        if k.startswith('src.'):
                            del sys.modules[k]
                            
            engine = None
            if engine_name == "edge":
                from src.engines.edge import EdgeEngine
                engine = EdgeEngine(voice=voice_name)
                engine.generate(van_ban, out_wav, voice=voice_name)
            elif engine_name == "piper":
                from src.engines.piper import PiperEngine
                model_path = os.path.abspath(os.path.join(project_root, "models", "piper", voice_name))
                engine = PiperEngine(model_path=model_path)
                engine.generate(van_ban, out_wav, voice=voice_name)
            elif engine_name == "kokoro":
                from src.engines.kokoro import KokoroEngine
                engine = KokoroEngine()
                engine.generate(van_ban, out_wav, voice=voice_name)
            elif engine_name == "vieneu":
                from src.engines.vieneu import VieNeuEngine
                engine = VieNeuEngine()
                if voice_name in ["v3turbo", "standard"]:
                    v_name = "Ngọc Lan"
                    v_mode = voice_name
                elif voice_name and "|" in voice_name:
                    v_name, v_mode = voice_name.split("|")
                else:
                    v_name = "Ngọc Lan"
                    v_mode = "v3turbo"
                engine.generate(van_ban, out_wav, voice=v_name, vieneu_mode=v_mode)
            else:
                raise RuntimeError(f"Engine {engine_name} không hỗ trợ cho đọc văn bản đơn lẻ.")
                
            if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
                raise RuntimeError("Tạo giọng nói thất bại.")
                
            log_json("autosub_done", {"output": os.path.abspath(out_wav), "voiceover": os.path.abspath(out_wav), "doc_van_ban_only": True})
            sys.exit(0)

        if args.tao_anh_only:
            from app.services.storytelling.models import StoryContext
            from app.services.storytelling.image_generator import StorytellingPipeline
            import time

            try:
                import torch
            except ImportError:
                log_json("autosub_error", {"error": "Lỗi: Không tìm thấy thư viện torch. Vui lòng cài đặt (pip install torch diffusers)."})
                sys.exit(1)

            try:
                import diffusers
            except ImportError:
                log_json("autosub_error", {"error": "Lỗi: Không tìm thấy thư viện diffusers. Vui lòng cài đặt."})
                sys.exit(1)

            ctx = StoryContext(story_name="QuickImageGen", story_slug="quick_image_gen",
                               genre="", checkpoint=args.anh_model or "stablediffusionapi/anything-v5")
            pipeline = StorytellingPipeline(context=ctx)
            
            try:
                pipeline.warmup(args.anh_steps, args.anh_guidance)
                
                images = []
                seeds = []
                for i in range(args.anh_so):
                    current_seed = args.anh_seed if args.anh_seed != -1 else -1
                    if current_seed != -1 and i > 0:
                        current_seed += i
                        
                    img, final_seed = pipeline.generate_draft(
                        prompt=args.anh_prompt,
                        negative_prompt=args.anh_negative,
                        face_embedding=None,
                        face_image=None,
                        seed=current_seed,
                        width=args.anh_rong,
                        height=args.anh_cao,
                        num_steps=args.anh_steps,
                        guidance_scale=args.anh_guidance
                    )
                    
                    stamp = int(time.time())
                    os.makedirs(args.output_dir, exist_ok=True)
                    out_path = os.path.join(args.output_dir, f"anh_{stamp}_{i}_seed{final_seed}.png")
                    img.save(out_path)
                    
                    images.append(out_path)
                    seeds.append(final_seed)
                    
                    log_json("autosub_progress", {
                        "message": f"Đã tạo ảnh {i+1}/{args.anh_so}",
                        "percent": int(((i+1)/args.anh_so) * 100)
                    })
                    
                log_json("autosub_done", {
                    "images": images,
                    "seeds": seeds,
                    "warnings": pipeline.warnings if hasattr(pipeline, "warnings") else []
                })
            except Exception as e:
                log_json("autosub_error", {"error": f"Lỗi tạo ảnh: {e}"})
                sys.exit(1)
            finally:
                if hasattr(pipeline, "release"):
                    pipeline.release()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            sys.exit(0)

        if getattr(args, "minh_hoa_only", False):
            from app.config import config
            if args.llm_api_key:
                config.set_app_override("openai_api_key", args.llm_api_key)
            if args.llm_base_url:
                config.set_app_override("openai_base_url", args.llm_base_url)
            if args.llm_model:
                config.set_app_override("openai_model", args.llm_model)
                
            if args.pexels_api_key: config.set_app_override("pexels_api_keys", args.pexels_api_key)
            if args.pixabay_api_key: config.set_app_override("pixabay_api_keys", args.pixabay_api_key)
            if args.coverr_api_key: config.set_app_override("coverr_api_keys", args.coverr_api_key)
                
            from app.services.llm import extract_search_terms
            from app.services.material import search_videos_pexels, search_videos_pixabay, search_videos_coverr, save_video
            from app.services.material import VideoAspect
            
            try:
                # SRT đã được editor gộp thành đoạn ≥ 3 s — chỉ đọc (adapter không import được gói orchestrator của repo cha).
                doan = _doc_srt_don_gian(args.source_srt)
                
                kq_doan = []
                nguon = args.minh_hoa_nguon
                ti_le = VideoAspect.portrait if args.minh_hoa_ti_le == "9:16" else (VideoAspect.landscape if args.minh_hoa_ti_le == "16:9" else VideoAspect.square)
                
                for i, d in enumerate(doan):
                    log_json("autosub_progress", {"percent": int(i / len(doan) * 100), "message": f"Tìm minh hoạ đoạn {i+1}/{len(doan)}"})
                    tu_khoa = extract_search_terms(d["text"], amount=2)
                    tu_khoa_str = " ".join(tu_khoa) if tu_khoa else "video"
                    
                    vid_info_list = []
                    if nguon == "pexels":
                        vid_info_list = search_videos_pexels(tu_khoa_str, minimum_duration=int(d["t_ra"] - d["t_vao"]), video_aspect=ti_le)
                    elif nguon == "pixabay":
                        vid_info_list = search_videos_pixabay(tu_khoa_str, minimum_duration=int(d["t_ra"] - d["t_vao"]), video_aspect=ti_le)
                    elif nguon == "coverr":
                        vid_info_list = search_videos_coverr(tu_khoa_str, minimum_duration=int(d["t_ra"] - d["t_vao"]), video_aspect=ti_le)
                        
                    if vid_info_list:
                        vid_info = vid_info_list[0]
                        vid_file = save_video(vid_info.url, args.output_dir)
                        if vid_file:
                            kq_doan.append({
                                "t_vao": d["t_vao"],
                                "t_ra": d["t_ra"],
                                "file": vid_file,
                                "tu_khoa": tu_khoa_str
                            })
                            continue
                    log_json("autosub_warn", {"message": f"Không tìm thấy/tải được video cho: {tu_khoa_str}"})
                
                log_json("autosub_done", {"doan": kq_doan, "minh_hoa_only": True})
            except Exception as e:
                log_json("autosub_error", {"error": f"Lỗi minh hoạ: {e}"})
                sys.exit(1)
            sys.exit(0)

        
        # Lazy load heavy dependencies
        from app.config import config
        from app.utils import utils
        
        task_dir = utils.task_dir(task_id)

        # CB2: set key config.app["openai_*"] for translate_srt/dubbing (in-memory only).
        # set_app_override chu khong gan thang: load_config() chay lai nhieu lan
        # trong mot phien se ghi de gia tri rong tu config.toml.
        if args.llm_api_key:
            config.set_app_override("openai_api_key", args.llm_api_key)
        if args.llm_base_url:
            config.set_app_override("openai_base_url", args.llm_base_url)
        if args.llm_model:
            config.set_app_override("openai_model", args.llm_model)

        if args.whisper_model:
            config.set_whisper_override("model_size", args.whisper_model)
        if args.whisper_device:
            import torch
            dev = args.whisper_device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            config.set_whisper_override("device", dev)
            config.set_whisper_override("compute_type", "float16" if dev == "cuda" else "int8")

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
                
            def get_ocr_lang(src: str) -> str:
                mapping = {
                    "vietnamese": "vi", "english": "en", "chinese": "ch", "zh": "ch",
                    "japanese": "japan", "korean": "korean", "thai": "th",
                    "indonesian": "id", "latin": "latin", "french": "fr",
                    "german": "german", "spanish": "es", "russian": "ru"
                }
                sl = src.strip().lower()
                if sl in mapping:
                    return mapping[sl]
                return "latin"

            ocr_lang = get_ocr_lang(args.source_lang)
            if ocr_lang == "latin" and args.source_lang.strip().lower() not in ["latin", "indonesian"]:
                log_json("autosub_warn", {"message": f"Ngôn ngữ '{args.source_lang}' không hỗ trợ OCR tĩnh, dùng 'latin' để giữ chữ cái."})
            
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

        # Tách giọng và Làm nét
        if getattr(args, "tach_giong_only", False):
            from app.services.audio_cleaner import isolate_vocals
            log_json("autosub_progress", {"message": "Bắt đầu tách giọng (Demucs)...", "percent": 5})
            # Demucs tự đọc mp4 cần ffmpeg trên PATH (máy đích không có) → trích .wav stereo 44.1 kHz bằng
            # ffmpeg của imageio_ffmpeg trước; giữ stereo/44.1 kHz vì bản nhạc nền còn được dùng lại trên timeline.
            import imageio_ffmpeg
            import subprocess
            wav_vao = os.path.join(task_dir, "tach_giong_nguon.wav")
            res = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", video_path, "-vn", "-ac", "2",
                                  "-ar", "44100", "-acodec", "pcm_s16le", wav_vao], capture_output=True, text=True)
            if res.returncode != 0 or not os.path.exists(wav_vao):
                raise RuntimeError("Video không có luồng âm thanh để tách giọng.")
            vocals_path = isolate_vocals(wav_vao, task_dir)
            if vocals_path == wav_vao or not os.path.exists(vocals_path):
                raise RuntimeError("Tách giọng thất bại (Demucs lỗi).")
            no_vocals_path = vocals_path.replace("vocals.wav", "no_vocals.wav")
            audio_dir = args.audio_out_dir or args.output_dir
            os.makedirs(audio_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            dst_giong = os.path.join(audio_dir, f"{base_name}.giong.wav")
            idx = 1
            while os.path.exists(dst_giong):
                idx += 1
                dst_giong = os.path.join(audio_dir, f"{base_name}.giong({idx}).wav")
                
            dst_nhac = os.path.join(audio_dir, f"{base_name}.nhac.wav")
            if os.path.exists(no_vocals_path):
                idx_nhac = 1
                while os.path.exists(dst_nhac):
                    idx_nhac += 1
                    dst_nhac = os.path.join(audio_dir, f"{base_name}.nhac({idx_nhac}).wav")
            
            shutil.copy(vocals_path, dst_giong)
            if os.path.exists(no_vocals_path):
                shutil.copy(no_vocals_path, dst_nhac)
            log_json("autosub_done", {
                "output": os.path.abspath(dst_giong),
                "vocals": os.path.abspath(dst_giong),
                "no_vocals": os.path.abspath(dst_nhac) if os.path.exists(no_vocals_path) else "",
                "tach_giong_only": True
            })
            sys.exit(0)
            
        if getattr(args, "can_gio_only", False):
            from app.services.subtitle import WhisperModel
            from app.config import config as app_config
            import torch
            cg_model_size = args.whisper_model if args.whisper_model else app_config.whisper.get("model_size", "base")
            
            cg_device = args.whisper_device if args.whisper_device else app_config.whisper.get("device", "cpu")
            cg_compute_type = app_config.whisper.get("compute_type", "int8")
            
            if cg_device == "auto":
                cg_device = "cuda" if torch.cuda.is_available() else "cpu"
            if cg_device == "cuda":
                cg_compute_type = "float16"
            
            log_json("autosub_progress", {"message": "Bắt đầu chạy Whisper để căn giờ...", "percent": 5})
            cg_model = WhisperModel(model_size_or_path=cg_model_size, device=cg_device, compute_type=cg_compute_type)
            cg_segments, _ = cg_model.transcribe(
                video_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            cg_words = []
            for cg_seg in cg_segments:
                if getattr(cg_seg, "words", None):
                    for cg_w in cg_seg.words:
                        cg_words.append({"w": cg_w.word, "bd": cg_w.start, "kt": cg_w.end})
            
            cg_out_json = os.path.join(args.output_dir, "words.json")
            with open(cg_out_json, "w", encoding="utf-8") as f:
                json.dump(cg_words, f, ensure_ascii=False)
                
            log_json("autosub_done", {"words": os.path.abspath(cg_out_json), "can_gio_only": True})
            sys.exit(0)

        if getattr(args, "lam_net_only", False):
            # Cần patch sys.path để src module hoạt động
            mc_root_parent = os.path.abspath(os.path.join(mc_root, "..", ".."))
            if mc_root_parent not in sys.path:
                sys.path.insert(0, mc_root_parent)
            from src.utils.video_processor import process_animation_video
            
            kieu_map = {
                "nhanh": "Làm nét nhanh (FFmpeg CAS)",
                "ai": "RealESRGAN_x4plus_anime_6B",
                "ai_video": "AnimeVideo-V3"
            }
            upscale_method = kieu_map.get(args.lam_net_kieu, "Làm nét nhanh (FFmpeg CAS)")
            
            if args.lam_net_kieu == "ai_video":
                # Kiểm tra model có tồn tại không
                model_path = os.path.join(mc_root, "models", "realesrgan", "realesr-animevideov3.pth")
                if not os.path.exists(model_path):
                    raise RuntimeError("Không tìm thấy model realesr-animevideov3 trong AIVoice/apps/MediaComposer/models/realesrgan/")

            log_json("autosub_progress", {"message": f"Bắt đầu làm nét ({upscale_method})...", "percent": 5})
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            dst = os.path.join(args.output_dir, f"{base_name}_net.mp4")
            
            idx = 1
            while os.path.exists(dst):
                idx += 1
                dst = os.path.join(args.output_dir, f"{base_name}_net({idx}).mp4")
                
            temp_dir = os.path.join(task_dir, "lam_net")
            
            def log_cb(msg):
                log_json("autosub_progress", {"message": msg})
                
            success = process_animation_video(
                input_path=video_path,
                output_path=dst,
                temp_dir=temp_dir,
                upscale_method=upscale_method,
                resolution=args.lam_net_do_phan_giai,
                log_callback=log_cb
            )
            if not success or not os.path.exists(dst):
                raise RuntimeError("Làm nét thất bại.")
            log_json("autosub_done", {
                "output": os.path.abspath(dst),
                "lam_net_only": True
            })
            sys.exit(0)

        # Run translation workflow
        from app.services.composer import composer
        
        log_json("autosub_progress", {"message": "Bắt đầu chạy workflow tạo phụ đề và lồng tiếng...", "percent": 10})
        
        clean_audio_flag = args.clean_audio if args.sub_source == "whisper" else False
        
        workflow_result = composer.run_translation_workflow(
            task_id=task_id,
            video_path=video_path,
            source_lang=args.source_lang,
            burn_method=args.burn_method,
            enable_voiceover=args.enable_voiceover or args.voiceover_only,
            voiceover_only=args.voiceover_only,
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

        if args.voiceover_only:
            audio_dir = args.audio_out_dir or args.output_dir
            os.makedirs(audio_dir, exist_ok=True)
            
            engine_str = args.tts_engine or "tts"
            # Giọng clone là đường dẫn file mẫu → chỉ lấy tên; bỏ ký tự Windows cấm trong tên file (`:` của ổ đĩa…).
            voice_str = os.path.splitext(os.path.basename(args.tts_voice or ""))[0] if os.path.sep in (args.tts_voice or "") or "/" in (args.tts_voice or "") else (args.tts_voice or "")
            voice_str = re.sub(r'[<>:"/\\|?*\s]+', "-", voice_str).strip("-")
            suffix = f"{engine_str}-{voice_str}" if voice_str else engine_str
            base_dst_name = f"{base_name}.{lang_tag(args.target_lang)}.{suffix}.long_tieng"
            dst = os.path.join(audio_dir, f"{base_dst_name}.wav")
            idx = 1
            while os.path.exists(dst):
                idx += 1
                dst = os.path.join(audio_dir, f"{base_dst_name}({idx}).wav")
                
            shutil.copy(workflow_result, dst)
            log_json("autosub_done", {"output": os.path.abspath(dst), "voiceover": os.path.abspath(dst),
                                      "voiceover_only": True, **exported})
            sys.exit(0)

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
