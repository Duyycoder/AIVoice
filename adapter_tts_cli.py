import os
import sys
import json
import argparse
import gc

# Add current folder to sys.path to resolve src package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import process_single_file

def log_json(event: str, data: dict):
    """Outputs progress log as a JSON string to stdout."""
    print(json.dumps({"event": event, **data}, ensure_ascii=False))
    sys.stdout.flush()

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main():
    parser = argparse.ArgumentParser(description="Non-interactive CLI Adapter for AIVoice TTS")
    parser.add_argument("--preset", default=None, help="Preset name from configs/tts_presets.json")
    parser.add_argument("--input", help="Path to input text file (.txt/.md)")
    parser.add_argument("--input-dir", help="Path to input directory of chapters for batch mode")
    parser.add_argument("--output", help="Path to save final audio output file (.wav)")
    parser.add_argument("--output-dir", help="Path to save batch audio outputs")

    # Overrides for preset / defaults
    parser.add_argument("--engine", help="TTS Engine (edge/piper/clone/kokoro/vieneu)")
    parser.add_argument("--voice", help="Voice model / gender key")
    parser.add_argument("--speed", type=float, help="Speech speed multiplier")
    parser.add_argument("--model", help="Path to local TTS model file (.onnx / folder)")
    parser.add_argument("--ref-audio", help="Path to reference audio for voice cloning")
    parser.add_argument("--phonemize", action="store_true", default=None, help="Enable Vietnamese phonemizer")
    parser.add_argument("--no-phonemize", action="store_false", dest="phonemize", help="Disable Vietnamese phonemizer")
    parser.add_argument("--normalize", action="store_true", default=None, help="Enable LUFS volume normalization")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="Disable LUFS volume normalization")
    parser.add_argument("--target-lufs", type=float, help="Target LUFS volume level (e.g. -14.0)")
    parser.add_argument("--fade-in", type=float, help="Fade-in duration in seconds")
    parser.add_argument("--fade-out", type=float, help="Fade-out duration in seconds")
    parser.add_argument("--silence-duration", type=float, help="Silence gap between segments in seconds")
    parser.add_argument("--device", default="cuda", help="Computation device (cuda/cpu)")
    
    # Advanced tweaks
    parser.add_argument("--use-cache", action="store_true", default=None)
    parser.add_argument("--no-cache", action="store_false", dest="use_cache")
    parser.add_argument("--cache-threshold", type=float, default=None)
    parser.add_argument("--vieneu-mode", default=None)
    parser.add_argument("--vieneu-emotion", default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()

    # Load presets
    presets = {}
    preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "tts_presets.json")
    if os.path.exists(preset_path):
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
        except Exception as e:
            log_json("preset_warn", {"error": f"Failed to load presets: {e}"})

    # Initialize configuration dictionary with default settings
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.json")
    config_data = {}
    if os.path.exists(default_config_path):
        try:
            with open(default_config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            pass

    # Merge preset if specified
    if args.preset:
        if args.preset in presets:
            config_data.update(presets[args.preset])
            log_json("preset_loaded", {"preset": args.preset})
        else:
            log_json("preset_warn", {"message": f"Preset '{args.preset}' not found. Using defaults."})

    # Merge command line arguments overrides
    override_fields = [
        "engine", "voice", "speed", "model", "ref_audio", "phonemize", 
        "normalize", "target_lufs", "fade_in", "fade_out", "silence_duration", "device",
        "use_cache", "cache_threshold", "vieneu_mode", "vieneu_emotion", "temperature"
    ]
    for field in override_fields:
        arg_val = getattr(args, field, None)
        if arg_val is not None:
            config_data[field] = arg_val

    # Ensure clean default values for unspecified/None fields
    config_data.setdefault("engine", "edge")
    config_data.setdefault("voice", "vi-VN-NamMinhNeural")
    config_data.setdefault("speed", 1.0)
    config_data.setdefault("model", None)
    config_data.setdefault("ref_audio", None)
    config_data.setdefault("phonemize", False)
    config_data.setdefault("normalize", True)
    config_data.setdefault("target_lufs", -14.0)
    config_data.setdefault("fade_in", 0.1)
    config_data.setdefault("fade_out", 0.1)
    config_data.setdefault("silence_duration", 0.3)
    config_data.setdefault("device", "cuda")
    config_data.setdefault("use_cache", False)
    config_data.setdefault("cache_threshold", 0.95)
    config_data.setdefault("vieneu_mode", "v3turbo")
    config_data.setdefault("vieneu_emotion", "natural")
    config_data.setdefault("temperature", 0.3)
    config_data.setdefault("use_fp16", True)
    config_data.setdefault("use_tf32", True)
    config_data.setdefault("max_words", 30)

    # Convert to standard Namespace for src/main.py consumption
    runner_args = Namespace(**config_data)

    # Initialize Engine Plugin exactly like src/main.py does
    engine = None
    if runner_args.engine == "piper":
        from src.engines.piper import PiperEngine
        engine = PiperEngine(runner_args.model)
    elif runner_args.engine == "edge":
        from src.engines.edge import EdgeEngine
        engine = EdgeEngine(runner_args.voice)
    elif runner_args.engine == "clone":
        if not runner_args.ref_audio:
            log_json("tts_error", {"message": "ref_audio is required for the CloneEngine"})
            sys.exit(1)
        from src.engines.clone import CloneEngine
        engine = CloneEngine(runner_args.model)
    elif runner_args.engine == "kokoro":
        from src.engines.kokoro import KokoroEngine
        engine = KokoroEngine(runner_args.model)
    elif runner_args.engine == "vieneu":
        from src.engines.vieneu import VieNeuEngine
        engine = VieNeuEngine(runner_args.model)
    else:
        log_json("tts_error", {"message": f"Unsupported tts engine: {runner_args.engine}"})
        sys.exit(1)

    try:
        # Determine mode: batch vs single file
        if args.input_dir:
            input_dir = os.path.abspath(args.input_dir)
            output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(input_dir, "audio")
            
            log_json("tts_batch_start", {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "engine": runner_args.engine
            })
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Logic "Tiếp tục": Lọc file [VI] và bỏ qua nếu đã có audio
            all_files = os.listdir(input_dir)
            if output_dir != input_dir and os.path.isdir(output_dir):
                all_audio_files = all_files + os.listdir(output_dir)
            else:
                all_audio_files = all_files
            audio_prefixes = {
                os.path.splitext(f)[0] for f in all_audio_files
                if f.lower().endswith((".mp3", ".wav"))
            }
            # Chương đã có audio (phần tên trước " - [VI] ") — để không đọc lại chương
            # khi tồn tại nhiều bản dịch trùng lặp với tiêu đề khác nhau
            audio_chapter_prefixes = {
                p.split(" - [VI] ")[0] for p in audio_prefixes if " - [VI] " in p
            }

            input_files = []
            seen_chapters = set()
            # Đếm riêng số chương HỢP LỆ để phân biệt hai tình huống cùng cho ra
            # danh sách rỗng: "đã đọc hết rồi" (bình thường) và "không có gì để
            # đọc" (lỗi cấu hình - phải dừng, đừng để bước sau lãnh thay).
            candidates = 0
            for f in sorted(all_files):
                if not (f.lower().endswith((".md", ".txt")) and " - [VI] " in f):
                    continue
                candidates += 1
                base_name = os.path.splitext(f)[0]
                chapter = base_name.split(" - [VI] ")[0]
                if base_name in audio_prefixes or chapter in audio_chapter_prefixes:
                    continue
                if chapter in seen_chapters:
                    log_json("tts_file_skip", {
                        "file": f,
                        "reason": f"Chương '{chapter}' có nhiều bản dịch [VI]; chỉ đọc bản đầu tiên."
                    })
                    continue
                seen_chapters.add(chapter)
                input_files.append(f)

            if not input_files:
                if candidates:
                    log_json("tts_batch_warn", {
                        "message": f"Cả {candidates} chương trong {input_dir} đã có audio — không còn gì để đọc."
                    })
                    sys.exit(0)
                log_json("tts_error", {"message": (
                    f"Không có tệp .md/.txt nào mang dấu ' - [VI] ' trong {input_dir}. "
                    "Thư mục chỉ có bản gốc chưa dịch, hoặc tên tệp chương sai quy tắc — "
                    "sửa bằng: python scripts/fix_ten_chuong.py <thư mục raw>"
                )})
                sys.exit(1)

            for idx, filename in enumerate(input_files, 1):
                file_path = os.path.join(input_dir, filename)
                name_without_ext = os.path.splitext(filename)[0].strip()
                file_output_path = os.path.join(output_dir, f"{name_without_ext}.wav")
                
                log_json("tts_file_start", {
                    "index": idx,
                    "total": len(input_files),
                    "file": filename
                })

                result = process_single_file(file_path, file_output_path, engine, runner_args)
                if result.get("status") == "SUCCESS":
                    log_json("tts_file_success", {
                        "index": idx,
                        "file": filename,
                        "output": file_output_path,
                        "duration_s": result.get("duration_s")
                    })
                else:
                    log_json("tts_file_failed", {"index": idx, "file": filename, "status": result.get("status")})

            log_json("tts_batch_completed", {"status": "success"})

        elif args.input:
            input_path = os.path.abspath(args.input)
            output_path = os.path.abspath(args.output) if args.output else os.path.splitext(input_path)[0] + ".wav"
            
            log_json("tts_start", {
                "input": input_path,
                "output": output_path,
                "engine": runner_args.engine,
                "voice": runner_args.voice
            })

            result = process_single_file(input_path, output_path, engine, runner_args)
            if result.get("status") == "SUCCESS":
                log_json("tts_success", {
                    "output": result.get("output"),
                    "duration_s": result.get("duration_s")
                })
            else:
                log_json("tts_failed", {"status": result.get("status")})
                sys.exit(1)
        else:
            log_json("tts_error", {"message": "Either --input or --input-dir must be provided."})
            sys.exit(1)

    except Exception as e:
        log_json("tts_error", {"error": str(e)})
        sys.exit(1)
    finally:
        # Crucial VRAM and RAM cleanup to release hardware resources after execution (Gap5)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        log_json("hardware_released", {"status": "success"})

if __name__ == "__main__":
    main()
