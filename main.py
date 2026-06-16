import argparse
import os
import sys
import tempfile
import time
import glob
import json

# Add PyTorch's bundled CUDA/cuDNN DLLs to Windows DLL search path
# so that onnxruntime-gpu can find them without system-wide CUDA installation.
if sys.platform == "win32":
    try:
        import torch
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib_dir):
            os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib_dir)
    except ImportError:
        pass

def process_single_file(input_path: str, output_path: str, engine, args) -> dict:
    """Process a single .md file through the TTS pipeline.
    
    Returns a dict with status info for the summary table.
    """
    result = {
        "input": os.path.basename(input_path),
        "output": output_path,
        "chunks": 0,
        "status": "FAILED",
        "duration_s": 0.0,
    }
    
    start_time = time.time()
    
    # Read input file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        import unicodedata
        raw_text = unicodedata.normalize("NFC", raw_text)
    except Exception as e:
        print(f"Error reading input file '{input_path}': {e}", file=sys.stderr)
        return result
        
    # Apply local AI text spicing if requested
    if args.spice_text:
        from src.utils.local_ai_spice import add_spice_to_text_local
        print("Adding AI emotion/spice to raw text using local LLM...")
        try:
            raw_text = add_spice_to_text_local(raw_text, style="teu_tao", model_path=args.llm_model)
        except Exception as e:
            print(f"Error adding AI emotion/spice: {e}", file=sys.stderr)
            result["status"] = "FAILED (AI Spice Error)"
            return result
        
    # Process text
    from src.utils.text import clean_markdown, chunk_text
    cleaned_text = clean_markdown(raw_text)
    chunks = chunk_text(cleaned_text)
    
    if not chunks:
        print(f"Warning: No text content found after cleaning '{input_path}'. Skipping.")
        result["status"] = "SKIPPED (empty)"
        return result
        
    # Apply Vietnamese Phonemizer if requested
    if args.phonemize:
        from src.utils.phoneme import phonemize_vietnamese
        print("Applying Vietnamese phonemization to text chunks...")
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            phonemized = phonemize_vietnamese(chunk)
            processed_chunks.append(phonemized)
        chunks = processed_chunks
        
    result["chunks"] = len(chunks)
    
    # Generate segments
    print(f"\nProcessing {len(chunks)} text chunks from '{os.path.basename(input_path)}'...")
    temp_files = []
    
    # Pre-create all temp files and tasks
    tasks = []
    for idx, chunk in enumerate(chunks):
        fd, temp_path = tempfile.mkstemp(suffix=f"_segment_{idx}.wav")
        os.close(fd)
        temp_files.append(temp_path)
        
        # Prepare kwargs
        kwargs = {}
        if args.model:
            kwargs["model"] = args.model
        if args.speed != 1.0:
            kwargs["speed"] = args.speed
        if args.voice:
            kwargs["voice"] = args.voice
        if args.ref_audio:
            kwargs["ref_audio"] = args.ref_audio
            
        tasks.append((idx, chunk, temp_path, kwargs))
        
    # Determine the number of workers based on the engine
    if args.engine == "clone":
        max_workers = 1
    elif args.engine == "piper":
        max_workers = 6
    else: # edge
        max_workers = 3

        
    success_map = {}
    
    def worker(task):
        idx, chunk, temp_path, kwargs = task
        preview_text = chunk[:40].replace('\n', ' ')
        try:
            success = engine.generate(chunk, temp_path, **kwargs)
            if success:
                print(f" -> Generated segment {idx+1}/{len(chunks)}: '{preview_text}...'")
            else:
                print(f"Error: Failed to generate audio for segment {idx+1}/{len(chunks)}: '{preview_text}...'", file=sys.stderr)
            return idx, success
        except Exception as e:
            print(f"Error: Exception during segment {idx+1}/{len(chunks)} generation: {e}", file=sys.stderr)
            return idx, False
            
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            idx, success = future.result()
            success_map[idx] = success
            
    # Check if all segments succeeded
    all_success = all(success_map.get(i, False) for i in range(len(chunks)))
    
    if not all_success:
        # Clean up temp files
        for p in temp_files:
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        return result
            
    # Concatenate and save
    from src.utils.audio import concatenate_wavs, apply_audio_post_processing
    
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"Concatenating {len(temp_files)} segments into final audio...")
    success = concatenate_wavs(temp_files, output_path, silence_duration=args.silence_duration)
    
    # Apply audio post-processing if successful
    if success:
        if args.normalize or args.fade_in > 0 or args.fade_out > 0:
            print(f"Applying post-processing (Normalize to LUFS: {args.target_lufs if args.normalize else 'No'}, Fade In: {args.fade_in}s, Fade Out: {args.fade_out}s)...")
            apply_audio_post_processing(
                output_path,
                target_lufs=args.target_lufs if args.normalize else None,
                fade_in_duration=args.fade_in,
                fade_out_duration=args.fade_out
            )
        
        result["status"] = "SUCCESS"
        print(f"SUCCESS: Audio saved to: {output_path}")
        
        # Apply RVC voice conversion if model is specified
        if args.rvc_model:
            rvc_output_path = os.path.splitext(output_path)[0] + "_rvc.wav"
            print("Applying RVC voice conversion (Voice-to-Voice)...")
            from src.engines.rvc_engine import apply_rvc
            rvc_success = apply_rvc(
                input_wav_path=output_path,
                output_wav_path=rvc_output_path,
                model_path=args.rvc_model,
                index_path=args.rvc_index,
                pitch_shift=args.rvc_pitch
            )
            if rvc_success:
                result["output"] = rvc_output_path
                print(f"SUCCESS: RVC audio saved to: {rvc_output_path}")
            else:
                print("Error: RVC voice conversion failed.", file=sys.stderr)
                result["status"] = "FAILED (RVC Error)"
    else:
        print(f"Error: Audio concatenation failed for '{input_path}'.", file=sys.stderr)
        
    elapsed = time.time() - start_time
    result["duration_s"] = elapsed
    return result


def process_single_audio_rvc(input_path: str, output_path: str, args) -> dict:
    """Process a single audio recording file through the RVC pipeline.
    
    Returns a dict with status info for the summary table.
    """
    result = {
        "input": os.path.basename(input_path),
        "output": output_path,
        "chunks": 1,
        "status": "FAILED",
        "duration_s": 0.0,
    }
    
    start_time = time.time()
    
    if not args.rvc_model:
        print("Error: --rvc_model is required for RVC voice conversion engine.", file=sys.stderr)
        return result
        
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"\nProcessing audio file '{os.path.basename(input_path)}' with RVC...")
    print("Applying RVC voice conversion (Voice-to-Voice)...")
    from src.engines.rvc_engine import apply_rvc
    
    success = apply_rvc(
        input_wav_path=input_path,
        output_wav_path=output_path,
        model_path=args.rvc_model,
        index_path=args.rvc_index,
        pitch_shift=args.rvc_pitch
    )
    
    if success:
        # Apply audio post-processing if successful
        from src.utils.audio import apply_audio_post_processing
        if args.normalize or args.fade_in > 0 or args.fade_out > 0:
            print(f"Applying post-processing (Normalize to LUFS: {args.target_lufs if args.normalize else 'No'}, Fade In: {args.fade_in}s, Fade Out: {args.fade_out}s)...")
            apply_audio_post_processing(
                output_path,
                target_lufs=args.target_lufs if args.normalize else None,
                fade_in_duration=args.fade_in,
                fade_out_duration=args.fade_out
            )
        result["status"] = "SUCCESS"
        print(f"SUCCESS: Audio saved to: {output_path}")
    else:
        print("Error: RVC voice conversion failed.", file=sys.stderr)
        result["status"] = "FAILED (RVC Error)"
        
    elapsed = time.time() - start_time
    result["duration_s"] = elapsed
    return result


def print_summary_table(results: list[dict]):
    """Print a formatted summary table of batch processing results."""
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    
    if not results:
        print("No files were processed.")
        print("=" * 80)
        return
        
    # Column widths
    col_file = max(len("Input File"), max(len(r["input"]) for r in results))
    col_chunks = len("Chunks")
    col_time = len("Time (s)")
    col_status = max(len("Status"), max(len(r["status"]) for r in results))
    
    # Header
    header = f"{'Input File':<{col_file}}  {'Chunks':>{col_chunks}}  {'Time (s)':>{col_time}}  {'Status':<{col_status}}"
    print(header)
    print("-" * len(header))
    
    # Rows
    success_count = 0
    for r in results:
        time_str = f"{r['duration_s']:.1f}"
        print(f"{r['input']:<{col_file}}  {r['chunks']:>{col_chunks}}  {time_str:>{col_time}}  {r['status']:<{col_status}}")
        if r["status"] == "SUCCESS":
            success_count += 1
            
    print("-" * len(header))
    print(f"Total: {len(results)} files | Success: {success_count} | Failed: {len(results) - success_count}")
    print("=" * 80)


def run_interactive_wizard(config_data: dict) -> list[str]:
    """Run an interactive wizard to prompt the user for options and return sys.argv-like list of arguments."""
    print("\n" + "=" * 60)
    print("      CẤU HÌNH CHẠY AIVOICE TTS (INTERACTIVE MODE)")
    print("=" * 60)
    
    cmd_args = ["main.py"]
    
    # 1. Select Engine / Mode
    print("\n[1] Chọn Engine / Chế độ chạy:")
    print("  1. edge (TTS - Microsoft Edge Online)")
    print("  2. piper (Piper ONNX - Offline)")
    print("  3. clone (Voice Cloning XTTSv2 - Offline)")
    print("  4. rvc (Voice-to-Voice RVC - Offline)")
    print("  5. batch (Xử lý hàng loạt toàn bộ file trong thư mục)")
    
    d_engine = config_data.get("engine", "edge")
    engine_choices = {
        "1": "edge", "edge": "edge",
        "2": "piper", "piper": "piper",
        "3": "clone", "clone": "clone",
        "4": "rvc", "rvc": "rvc",
        "5": "batch", "batch": "batch"
    }
    
    d_engine_idx = "1"
    if d_engine == "piper": d_engine_idx = "2"
    elif d_engine == "clone": d_engine_idx = "3"
    elif d_engine == "rvc": d_engine_idx = "4"
    
    engine_sel = input(f"Chọn [Mặc định: {d_engine_idx} ({d_engine})]: ").strip().lower()
    if not engine_sel:
        engine_choice = d_engine
    else:
        engine_choice = engine_choices.get(engine_sel, d_engine)
        
    is_batch = False
    if engine_choice == "batch":
        is_batch = True
        print("\nChọn Engine để chạy Batch:")
        print("  1. edge (TTS - Microsoft Edge Online)")
        print("  2. piper (Piper ONNX - Offline)")
        print("  3. clone (Voice Cloning XTTSv2 - Offline)")
        print("  4. rvc (Voice-to-Voice RVC - Offline)")
        batch_engine_sel = input("Chọn [Mặc định: 1 (edge)]: ").strip()
        if batch_engine_sel == "2":
            engine_choice = "piper"
        elif batch_engine_sel == "3":
            engine_choice = "clone"
        elif batch_engine_sel == "4":
            engine_choice = "rvc"
        else:
            engine_choice = "edge"
            
    cmd_args.extend(["--engine", engine_choice])

    # 2. Select Model
    model_choice = None
    if engine_choice in ["piper", "clone"]:
        print("\n[2] Chọn Model:")
        models_dir = "models"
        available_models = []
        if os.path.exists(models_dir):
            if engine_choice == "piper":
                piper_dir = os.path.join(models_dir, "piper")
                if os.path.exists(piper_dir):
                    available_models = [os.path.join(piper_dir, f) for f in os.listdir(piper_dir) if f.endswith(".onnx")]
            elif engine_choice == "clone":
                xtts_path = os.path.join(models_dir, "xtts_v2")
                if os.path.exists(xtts_path):
                    available_models.append(xtts_path)
                for d in os.listdir(models_dir):
                    if d == "piper":
                        continue
                    d_path = os.path.join(models_dir, d)
                    if os.path.isdir(d_path) and d_path not in available_models:
                        available_models.append(d_path)
        
        for idx, m in enumerate(available_models):
            print(f"  {idx + 1}. {m}")
        print(f"  {len(available_models) + 1}. Tự nhập đường dẫn khác...")
        
        d_model = config_data.get("model")
        d_model_idx = ""
        if d_model:
            for idx, m in enumerate(available_models):
                if os.path.abspath(m) == os.path.abspath(d_model) or m == d_model:
                    d_model_idx = str(idx + 1)
                    break
        if not d_model_idx and available_models:
            d_model_idx = "1"
            d_model = available_models[0]
            
        default_prompt = f"Chọn [Mặc định: {d_model_idx} ({d_model})]" if d_model else "Chọn"
        model_sel = input(f"{default_prompt}: ").strip()
        
        if not model_sel:
            model_choice = d_model
        elif model_sel.isdigit() and 1 <= int(model_sel) <= len(available_models):
            model_choice = available_models[int(model_sel) - 1]
        else:
            if model_sel.isdigit() and int(model_sel) == len(available_models) + 1:
                model_choice = input("Nhập đường dẫn model: ").strip()
            else:
                model_choice = model_sel
                
        if model_choice:
            cmd_args.extend(["--model", model_choice])
            
    elif engine_choice == "rvc":
        print("\n[2] Chọn RVC Model:")
        rvc_dir = os.path.join("models", "rvc")
        available_rvc_models = []
        if os.path.exists(rvc_dir):
            available_rvc_models = [os.path.join(rvc_dir, f) for f in os.listdir(rvc_dir) if f.endswith(".pth")]
            
        for idx, m in enumerate(available_rvc_models):
            print(f"  {idx + 1}. {m}")
        print(f"  {len(available_rvc_models) + 1}. Tự nhập đường dẫn khác...")
        
        d_rvc_model = config_data.get("rvc_model", "models/rvc/ElevenLabs_Adam_FR.pth")
        d_model_idx = ""
        for idx, m in enumerate(available_rvc_models):
            if os.path.abspath(m) == os.path.abspath(d_rvc_model) or m == d_rvc_model:
                d_model_idx = str(idx + 1)
                break
        if not d_model_idx and available_rvc_models:
            d_model_idx = "1"
            d_rvc_model = available_rvc_models[0]
            
        default_prompt = f"Chọn [Mặc định: {d_model_idx} ({d_rvc_model})]" if d_rvc_model else "Chọn RVC model"
        model_sel = input(f"{default_prompt}: ").strip()
        
        if not model_sel:
            rvc_model_choice = d_rvc_model
        elif model_sel.isdigit() and 1 <= int(model_sel) <= len(available_rvc_models):
            rvc_model_choice = available_rvc_models[int(model_sel) - 1]
        else:
            if model_sel.isdigit() and int(model_sel) == len(available_rvc_models) + 1:
                rvc_model_choice = input("Nhập đường dẫn RVC model (.pth): ").strip()
            else:
                rvc_model_choice = model_sel
                
        if rvc_model_choice:
            cmd_args.extend(["--rvc_model", rvc_model_choice])
            
        # Select RVC Index
        print("\nChọn RVC Index (Tùy chọn):")
        available_indexes = []
        if os.path.exists(rvc_dir):
            available_indexes = [os.path.join(rvc_dir, f) for f in os.listdir(rvc_dir) if f.endswith(".index")]
            
        for idx, ind in enumerate(available_indexes):
            print(f"  {idx + 1}. {ind}")
        print(f"  {len(available_indexes) + 1}. Tự nhập đường dẫn index khác...")
        print(f"  {len(available_indexes) + 2}. Không sử dụng index")
        
        d_rvc_index = config_data.get("rvc_index", "")
        d_index_idx = ""
        if d_rvc_index:
            for idx, ind in enumerate(available_indexes):
                if os.path.abspath(ind) == os.path.abspath(d_rvc_index) or ind == d_rvc_index:
                    d_index_idx = str(idx + 1)
                    break
        if not d_index_idx and available_indexes:
            # Match by name if possible
            base_name_model = os.path.splitext(os.path.basename(rvc_model_choice))[0]
            for idx, ind in enumerate(available_indexes):
                if base_name_model in os.path.basename(ind):
                    d_index_idx = str(idx + 1)
                    d_rvc_index = ind
                    break
            if not d_index_idx:
                d_index_idx = str(len(available_indexes) + 2)
                d_rvc_index = ""
                
        default_prompt = f"Chọn [Mặc định: {d_index_idx} ({d_rvc_index if d_rvc_index else 'Không sử dụng'})]"
        index_sel = input(f"{default_prompt}: ").strip()
        
        if not index_sel:
            rvc_index_choice = d_rvc_index
        elif index_sel.isdigit() and 1 <= int(index_sel) <= len(available_indexes):
            rvc_index_choice = available_indexes[int(index_sel) - 1]
        elif index_sel.isdigit() and int(index_sel) == len(available_indexes) + 1:
            rvc_index_choice = input("Nhập đường dẫn RVC Index (.index): ").strip()
        else:
            rvc_index_choice = ""
            
        if rvc_index_choice:
            cmd_args.extend(["--rvc_index", rvc_index_choice])
            
        # Pitch shift
        d_rvc_pitch = config_data.get("rvc_pitch", 0)
        rvc_pitch_sel = input(f"\nNhập thông số pitch shift (Số nguyên, mặc định: {d_rvc_pitch}): ").strip()
        rvc_pitch_choice = int(rvc_pitch_sel) if rvc_pitch_sel else d_rvc_pitch
        cmd_args.extend(["--rvc_pitch", str(rvc_pitch_choice)])

    # 3. Select Input
    print("\n[3] Chọn Input:")
    AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
    
    if engine_choice == "rvc":
        if is_batch:
            d_input_dir = "data/recordings"
            if not os.path.exists(d_input_dir):
                os.makedirs(d_input_dir, exist_ok=True)
            input_dir_sel = input(f"Nhập đường dẫn thư mục chứa files ghi âm [Mặc định: {d_input_dir}]: ").strip()
            input_dir_choice = input_dir_sel if input_dir_sel else d_input_dir
            cmd_args.extend(["--input_dir", input_dir_choice])
        else:
            input_dir = "data/recordings"
            if not os.path.exists(input_dir):
                os.makedirs(input_dir, exist_ok=True)
                
            available_inputs = []
            for d in [input_dir, "data/inputs"]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.lower().endswith(AUDIO_EXTENSIONS):
                            available_inputs.append(os.path.join(d, f))
            
            if available_inputs:
                for idx, inp in enumerate(available_inputs):
                    print(f"  {idx + 1}. {inp}")
                print(f"  {len(available_inputs) + 1}. Tự nhập đường dẫn file khác...")
                
                d_input = available_inputs[0]
                d_input_idx = "1"
                default_prompt = f"Chọn [Mặc định: {d_input_idx} ({d_input})]"
                input_sel = input(f"{default_prompt}: ").strip()
                
                if not input_sel:
                    input_choice = d_input
                elif input_sel.isdigit() and 1 <= int(input_sel) <= len(available_inputs):
                    input_choice = available_inputs[int(input_sel) - 1]
                else:
                    if input_sel.isdigit() and int(input_sel) == len(available_inputs) + 1:
                        input_choice = input("Nhập đường dẫn file ghi âm: ").strip()
                    else:
                        input_choice = input_sel
            else:
                input_choice = input(f"Nhập đường dẫn file ghi âm (e.g. .mp3, .wav): ").strip()
                
            cmd_args.extend(["--input", input_choice])
    else:
        if is_batch:
            d_input_dir = "data/inputs"
            input_dir_sel = input(f"Nhập đường dẫn thư mục chứa files [Mặc định: {d_input_dir}]: ").strip()
            input_dir_choice = input_dir_sel if input_dir_sel else d_input_dir
            cmd_args.extend(["--input_dir", input_dir_choice])
        else:
            input_dir = "data/inputs"
            available_inputs = []
            if os.path.exists(input_dir):
                available_inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".md") or f.endswith(".txt")]
                
            for idx, inp in enumerate(available_inputs):
                print(f"  {idx + 1}. {inp}")
            print(f"  {len(available_inputs) + 1}. Tự nhập đường dẫn file khác...")
            
            d_input = available_inputs[0] if available_inputs else ""
            d_input_idx = "1" if available_inputs else ""
            
            default_prompt = f"Chọn [Mặc định: {d_input_idx} ({d_input})]" if d_input else "Nhập đường dẫn file input"
            input_sel = input(f"{default_prompt}: ").strip()
            
            if not input_sel:
                input_choice = d_input
            elif input_sel.isdigit() and 1 <= int(input_sel) <= len(available_inputs):
                input_choice = available_inputs[int(input_sel) - 1]
            else:
                if input_sel.isdigit() and int(input_sel) == len(available_inputs) + 1:
                    input_choice = input("Nhập đường dẫn file: ").strip()
                else:
                    input_choice = input_sel
                    
            cmd_args.extend(["--input", input_choice])

    # 4. Select Voice (giọng)
    if engine_choice != "rvc":
        print("\n[4] Chọn Giọng (Voice) / Ngôn ngữ:")
        if engine_choice == "edge":
            print("  1. vi-VN-NamMinhNeural (Nam Minh - Giọng nam)")
            print("  2. vi-VN-HoaiMyNeural (Hoài My - Giọng nữ)")
            print("  3. Tự nhập tên giọng nói khác...")
            d_voice = config_data.get("voice", "vi-VN-NamMinhNeural")
            d_voice_idx = "1" if d_voice == "vi-VN-NamMinhNeural" else ("2" if d_voice == "vi-VN-HoaiMyNeural" else "3")
            voice_sel = input(f"Chọn [Mặc định: {d_voice_idx} ({d_voice})]: ").strip()
            if not voice_sel:
                voice_choice = d_voice
            elif voice_sel == "1":
                voice_choice = "vi-VN-NamMinhNeural"
            elif voice_sel == "2":
                voice_choice = "vi-VN-HoaiMyNeural"
            else:
                if voice_sel == "3":
                    voice_choice = input("Nhập tên giọng nói Edge-TTS: ").strip()
                else:
                    voice_choice = voice_sel
            cmd_args.extend(["--voice", voice_choice])
            
        elif engine_choice == "clone":
            print("  1. vi (Tiếng Việt)")
            print("  2. en (Tiếng Anh)")
            print("  3. zh-cn (Tiếng Trung)")
            print("  4. ja (Tiếng Nhật)")
            print("  5. ko (Tiếng Hàn)")
            print("  6. fr (Tiếng Pháp)")
            print("  7. es (Tiếng Tây Ban Nha)")
            print("  8. de (Tiếng Đức)")
            print("  9. Tự nhập mã ngôn ngữ khác...")
            
            d_voice = config_data.get("voice", "en")
            lang_map = {
                "1": "vi", "2": "en", "3": "zh-cn", "4": "ja", 
                "5": "ko", "6": "fr", "7": "es", "8": "de"
            }
            
            d_voice_idx = "9"
            for idx, lcode in lang_map.items():
                if d_voice == lcode:
                    d_voice_idx = idx
                    break
                    
            voice_sel = input(f"Chọn [Mặc định: {d_voice_idx} ({d_voice})]: ").strip()
            if not voice_sel:
                voice_choice = d_voice
            elif voice_sel in lang_map:
                voice_choice = lang_map[voice_sel]
            elif voice_sel == "9":
                voice_choice = input("Nhập mã ngôn ngữ (e.g. ja, ko): ").strip()
            else:
                voice_choice = voice_sel
                
            cmd_args.extend(["--voice", voice_choice])
            
            # Scan for wav files in data/voices
            voices_dir = os.path.join("data", "voices")
            wav_files = []
            if os.path.exists(voices_dir):
                wav_files = [f for f in os.listdir(voices_dir) if f.endswith(".wav")]
            if wav_files:
                print("\nChọn File Âm Thanh Mẫu (Reference Audio):")
                for idx, w in enumerate(wav_files):
                    print(f"  {idx + 1}. {os.path.join(voices_dir, w)}")
                print(f"  {len(wav_files) + 1}. Tự nhập đường dẫn file khác...")
                
                d_ref_audio = config_data.get("ref_audio", "data/voices/ref_voice.wav")
                d_ref_idx = ""
                for idx, w in enumerate(wav_files):
                    full_path = os.path.join(voices_dir, w)
                    if full_path == d_ref_audio or w == d_ref_audio:
                        d_ref_idx = str(idx + 1)
                        break
                if not d_ref_idx:
                    d_ref_idx = "1"
                    d_ref_audio = os.path.join(voices_dir, wav_files[0])
                    
                ref_audio_sel = input(f"Chọn [Mặc định: {d_ref_idx} ({d_ref_audio})]: ").strip()
                if not ref_audio_sel:
                    ref_audio_choice = d_ref_audio
                elif ref_audio_sel.isdigit() and 1 <= int(ref_audio_sel) <= len(wav_files):
                    ref_audio_choice = os.path.join(voices_dir, wav_files[int(ref_audio_sel) - 1])
                else:
                    if ref_audio_sel.isdigit() and int(ref_audio_sel) == len(wav_files) + 1:
                        ref_audio_choice = input("Nhập đường dẫn file âm thanh mẫu: ").strip()
                    else:
                        ref_audio_choice = ref_audio_sel
            else:
                d_ref_audio = config_data.get("ref_audio", "data/voices/ref_voice.wav")
                ref_audio_sel = input(f"Nhập đường dẫn file âm thanh mẫu [Mặc định: {d_ref_audio}]: ").strip()
                ref_audio_choice = ref_audio_sel if ref_audio_sel else d_ref_audio
                
            cmd_args.extend(["--ref_audio", ref_audio_choice])
            
        elif engine_choice == "piper":
            print("  Gợi ý chọn Speaker ID (dành cho model nhiều giọng đọc - Multi-speaker):")
            print("    - Nhấn Enter nếu model là Single-speaker (Mặc định: 0)")
            print("    - Hoặc nhập số ID người nói (e.g. 0, 1, 2...)")
            d_voice = config_data.get("voice")
            if not d_voice or not str(d_voice).isdigit():
                d_voice = "0"
            voice_sel = input(f"Chọn Speaker ID [Mặc định: {d_voice}]: ").strip()
            voice_choice = voice_sel if voice_sel else d_voice
            cmd_args.extend(["--voice", voice_choice])

    # 5. Output file / folder
    print("\n[5] Tên đầu ra / Thư mục đầu ra:")
    if is_batch:
        d_output_dir = config_data.get("output_dir", "data/outputs")
        output_dir_sel = input(f"Nhập thư mục lưu file đầu ra [Mặc định: {d_output_dir}]: ").strip()
        output_dir_choice = output_dir_sel if output_dir_sel else d_output_dir
        cmd_args.extend(["--output_dir", output_dir_choice])
    else:
        input_basename = os.path.basename(cmd_args[cmd_args.index("--input") + 1])
        default_out_name = os.path.splitext(input_basename)[0]
        out_name_sel = input(f"Nhập tên file đầu ra (không cần đuôi .wav) [Mặc định: {default_out_name}]: ").strip()
        out_name = out_name_sel if out_name_sel else default_out_name
        
        d_output_dir = "data/outputs"
        output_dir_sel = input(f"Nhập thư mục lưu file đầu ra [Mặc định: {d_output_dir}]: ").strip()
        output_dir_choice = output_dir_sel if output_dir_sel else d_output_dir
        
        cmd_args.extend(["--output_dir", output_dir_choice])
        if out_name != default_out_name:
            cmd_args.extend(["--output_name", out_name])

    # 6. AI Emotion/Spice (Spice & Clone)
    if engine_choice != "rvc":
        print("\n[6] Thêm cảm xúc/hài hước bằng AI (Spice & Clone):")
        d_spice_text = config_data.get("spice_text", False)
        spice_sel = input(f"  Bạn có muốn thêm cảm xúc/hóm hỉnh vào văn bản bằng LLM cục bộ không? (y/n) [Mặc định: {'y' if d_spice_text else 'n'}]: ").strip().lower()
        
        spice_choice = None
        if spice_sel:
            spice_choice = spice_sel in ["y", "yes", "true"]
        
        actual_spice = spice_choice if spice_choice is not None else d_spice_text
        
        if actual_spice:
            if spice_choice is not None or d_spice_text:
                cmd_args.append("--spice_text")
            
            d_llm_model = config_data.get("llm_model", "")
            default_prompt = f" [Mặc định: {d_llm_model}]" if d_llm_model else ""
            llm_model_sel = input(f"  Nhập đường dẫn đến file model GGUF{default_prompt}: ").strip()
            llm_model_choice = llm_model_sel if llm_model_sel else d_llm_model
            if llm_model_choice:
                cmd_args.extend(["--llm_model", llm_model_choice])
        else:
            if spice_choice is not None:
                cmd_args.append("--no-spice_text")

    # 7. Voice-to-Voice (RVC) Post-Processing
    if engine_choice != "rvc":
        print("\n[7] Hậu kỳ Voice-to-Voice (RVC):")
        d_rvc_model = config_data.get("rvc_model", "")
        rvc_sel = input(f"  Bạn có muốn áp dụng RVC để đổi giọng (Voice-to-Voice) không? (y/n) [Mặc định: {'y' if d_rvc_model else 'n'}]: ").strip().lower()
        
        rvc_choice = None
        if rvc_sel:
            rvc_choice = rvc_sel in ["y", "yes", "true"]
            
        actual_rvc = rvc_choice if rvc_choice is not None else bool(d_rvc_model)
        
        if actual_rvc:
            default_prompt = f" [Mặc định: {d_rvc_model}]" if d_rvc_model else ""
            rvc_model_sel = input(f"  Nhập đường dẫn file RVC (.pth){default_prompt}: ").strip()
            rvc_model_choice = rvc_model_sel if rvc_model_sel else d_rvc_model
            if rvc_model_choice:
                cmd_args.extend(["--rvc_model", rvc_model_choice])
                
                d_rvc_index = config_data.get("rvc_index", "")
                default_index_prompt = f" [Mặc định: {d_rvc_index}]" if d_rvc_index else ""
                rvc_index_sel = input(f"  Nhập đường dẫn file RVC Index (.index - Tùy chọn){default_index_prompt}: ").strip()
                rvc_index_choice = rvc_index_sel if rvc_index_sel else d_rvc_index
                if rvc_index_choice:
                    cmd_args.extend(["--rvc_index", rvc_index_choice])
                    
                d_rvc_pitch = config_data.get("rvc_pitch", 0)
                rvc_pitch_sel = input(f"  Nhập thông số pitch shift (Số nguyên, mặc định: {d_rvc_pitch}): ").strip()
                rvc_pitch_choice = int(rvc_pitch_sel) if rvc_pitch_sel else d_rvc_pitch
                cmd_args.extend(["--rvc_pitch", str(rvc_pitch_choice)])
                
    # 8. Advanced options
    print("\n[8] Các thông số nâng cao (Ấn Enter để chọn giá trị mặc định):")
    
    if engine_choice != "rvc":
        d_speed = config_data.get("speed", 1.0)
        speed_sel = input(f"  Tốc độ đọc (speed) [Mặc định: {d_speed}]: ").strip()
        if speed_sel:
            cmd_args.extend(["--speed", speed_sel])
            
        d_phonemize = config_data.get("phonemize", False)
        phonemize_sel = input(f"  Bật chuyển tự phiên âm IPA (phonemize) (y/n) [Mặc định: {'y' if d_phonemize else 'n'}]: ").strip().lower()
        if phonemize_sel:
            if phonemize_sel in ["y", "yes", "true"]:
                cmd_args.append("--phonemize")
            else:
                cmd_args.append("--no-phonemize")
                
    d_normalize = config_data.get("normalize", True)
    normalize_sel = input(f"  Bật chuẩn hóa âm lượng (normalize) (y/n) [Mặc định: {'y' if d_normalize else 'n'}]: ").strip().lower()
    if normalize_sel:
        if normalize_sel in ["y", "yes", "true"]:
            cmd_args.append("--normalize")
        else:
            cmd_args.append("--no-normalize")
            
    d_target_lufs = config_data.get("target_lufs", -14.0)
    target_lufs_sel = input(f"  Độ lớn âm thanh mục tiêu (target LUFS) [Mặc định: {d_target_lufs}]: ").strip()
    if target_lufs_sel:
        cmd_args.extend(["--target_lufs", target_lufs_sel])
        
    d_fade_in = config_data.get("fade_in", 0.1)
    fade_in_sel = input(f"  Fade in (giây) [Mặc định: {d_fade_in}]: ").strip()
    if fade_in_sel:
        cmd_args.extend(["--fade_in", fade_in_sel])
        
    d_fade_out = config_data.get("fade_out", 0.1)
    fade_out_sel = input(f"  Fade out (giây) [Mặc định: {d_fade_out}]: ").strip()
    if fade_out_sel:
        cmd_args.extend(["--fade_out", fade_out_sel])
        
    if engine_choice != "rvc":
        d_silence_duration = config_data.get("silence_duration", 0.3)
        silence_duration_sel = input(f"  Khoảng lặng giữa các đoạn (silence duration) [Mặc định: {d_silence_duration}]: ").strip()
        if silence_duration_sel:
            cmd_args.extend(["--silence_duration", silence_duration_sel])
        
    print("\nLệnh chạy giả định:")
    print("python " + " ".join(cmd_args[1:]))
    print("=" * 60 + "\n")
    return cmd_args


def main():
    # Pre-parse config path to read default values from file first
    config_path = "configs/default.json"
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "--config":
            config_path = sys.argv[i+1]
            
    config_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            print(f"Loaded configuration defaults from '{config_path}'")
        except Exception as e:
            print(f"Warning: Failed to load config file '{config_path}': {e}", file=sys.stderr)
            
    # Check if run in interactive mode (no arguments, or only --config passed)
    is_interactive = False
    if len(sys.argv) == 1:
        is_interactive = True
    elif len(sys.argv) == 3 and sys.argv[1] == "--config":
        is_interactive = True
        
    if is_interactive:
        sys.argv = run_interactive_wizard(config_data)
        
    # Set default values from configuration file
    d_rvc_model = config_data.get("rvc_model", None)
    d_rvc_index = config_data.get("rvc_index", None)
    d_rvc_pitch = config_data.get("rvc_pitch", 0)
    d_spice_text = config_data.get("spice_text", False)
    d_llm_model = config_data.get("llm_model", None)
    d_engine = config_data.get("engine", None)
    d_model = config_data.get("model", None)
    d_ref_audio = config_data.get("ref_audio", "data/voices/ref_voice.wav")
    d_speed = config_data.get("speed", 1.0)
    d_voice = config_data.get("voice", None)
    d_phonemize = config_data.get("phonemize", False)
    d_normalize = config_data.get("normalize", True)
    d_target_lufs = config_data.get("target_lufs", -14.0)
    d_fade_in = config_data.get("fade_in", 0.1)
    d_fade_out = config_data.get("fade_out", 0.1)
    d_silence_duration = config_data.get("silence_duration", 0.3)

    parser = argparse.ArgumentParser(
        description="Extensible Single-Speaker Vietnamese TTS Framework"
    )
    
    # Input: either a single file or a directory (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to a single input .md file."
    )
    input_group.add_argument(
        "--input_dir",
        help="Path to a directory containing .md files for batch processing."
    )
    
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to configuration JSON file."
    )
    parser.add_argument(
        "--output_dir",
        default="data/outputs",
        help="Base output directory for batch processing (default: data/outputs/)."
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Custom name for the output audio file (without .wav, defaults to input base name)."
    )
    parser.add_argument(
        "--engine",
        required=(d_engine is None),
        default=d_engine,
        choices=["piper", "clone", "edge", "rvc"],
        help="Selection of the engine (piper, clone, edge, rvc)."
    )
    parser.add_argument(
        "--model",
        default=d_model,
        help="Local path to specific model weights."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=d_speed,
        help="Float value to control speech rate (e.g., 1.0, 1.2)."
    )
    parser.add_argument(
        "--voice",
        default=d_voice,
        help="Name of the voice (if applicable to the engine)."
    )
    parser.add_argument(
        "--ref_audio",
        default=d_ref_audio,
        help="Path to the reference .wav file (for CloneEngine)."
    )
    
    # Flags for Priority 3
    parser.add_argument(
        "--phonemize",
        action=argparse.BooleanOptionalAction,
        default=d_phonemize,
        help="Enable/disable converting Vietnamese text into IPA phonemes."
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=d_normalize,
        help="Enable/disable LUFS loudness normalization (-14 LUFS)."
    )
    parser.add_argument(
        "--target_lufs",
        type=float,
        default=d_target_lufs,
        help="Target loudness value in LUFS (default: -14.0)."
    )
    parser.add_argument(
        "--fade_in",
        type=float,
        default=d_fade_in,
        help="Duration of linear fade-in in seconds (default: 0.1)."
    )
    parser.add_argument(
        "--fade_out",
        type=float,
        default=d_fade_out,
        help="Duration of linear fade-out in seconds (default: 0.1)."
    )
    parser.add_argument(
        "--silence_duration",
        type=float,
        default=d_silence_duration,
        help="Silence duration in seconds between segments (default: 0.3)."
    )
    parser.add_argument(
        "--spice_text",
        action=argparse.BooleanOptionalAction,
        default=d_spice_text,
        help="Enable/disable rewriting text with local AI emotion/spice."
    )
    parser.add_argument(
        "--llm_model",
        default=d_llm_model,
        help="Path to the local GGUF LLM model."
    )
    parser.add_argument(
        "--rvc_model",
        default=d_rvc_model,
        help="Path to the RVC .pth model file."
    )
    parser.add_argument(
        "--rvc_index",
        default=d_rvc_index,
        help="Path to the RVC .index file (optional)."
    )
    parser.add_argument(
        "--rvc_pitch",
        type=int,
        default=d_rvc_pitch,
        help="Pitch shift for RVC (integer, default: 0)."
    )
    
    args = parser.parse_args()
    
    # In interactive mode, if the user chose not to enable RVC, ensure it is disabled
    # despite any defaults defined in configs/default.json.
    if is_interactive and "--rvc_model" not in sys.argv:
        args.rvc_model = None
        args.rvc_index = None
        
    # Resolve relative paths to absolute paths early, so they remain valid
    # regardless of any working directory changes during processing.
    if args.ref_audio:
        args.ref_audio = os.path.abspath(args.ref_audio)
    if args.model:
        args.model = os.path.abspath(args.model)
    if args.llm_model:
        args.llm_model = os.path.abspath(args.llm_model)
    if args.rvc_model:
        args.rvc_model = os.path.abspath(args.rvc_model)
    if args.rvc_index:
        args.rvc_index = os.path.abspath(args.rvc_index)
    
    # Initialize Engine Plugin
    engine = None
    if args.engine == "piper":
        from src.engines.piper import PiperEngine
        engine = PiperEngine(args.model)
    elif args.engine == "edge":
        from src.engines.edge import EdgeEngine
        engine = EdgeEngine(args.voice)
    elif args.engine == "clone":
        if not args.ref_audio:
            print("Error: --ref_audio is required for the CloneEngine", file=sys.stderr)
            sys.exit(1)
        try:
            from src.engines.clone import CloneEngine
            engine = CloneEngine(args.model)
        except ImportError:
            print("Error: The CloneEngine (XTTSv2) requires coqui-tts, which could not be loaded.", file=sys.stderr)
            sys.exit(1)
    elif args.engine == "rvc":
        if not args.rvc_model:
            print("Error: --rvc_model is required for RVC voice conversion engine.", file=sys.stderr)
            sys.exit(1)
            
    # Determine input files
    if args.input:
        # Single file mode
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        # Dynamic naming: output folder is named after input file
        input_base_name = os.path.splitext(os.path.basename(args.input))[0].strip()
        output_name = (args.output_name or input_base_name).strip()
        output_dir = os.path.join(args.output_dir, input_base_name)
        output_path = os.path.join(output_dir, f"{output_name}.wav")
        
        if args.engine == "rvc":
            result = process_single_audio_rvc(args.input, output_path, args)
        else:
            result = process_single_file(args.input, output_path, engine, args)
            
        if result["status"] != "SUCCESS":
            sys.exit(1)
    else:
        # Batch mode
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        is_rvc = (args.engine == "rvc")
        input_files = []
        extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg") if is_rvc else (".md", ".txt")
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith(extensions):
                input_files.append(filename)
        input_files.sort()
        
        if not input_files:
            if is_rvc:
                print(f"Error: No audio files found in '{args.input_dir}'.", file=sys.stderr)
            else:
                print(f"Error: No .md or .txt files found in '{args.input_dir}'.", file=sys.stderr)
            sys.exit(1)
            
        print(f"Found {len(input_files)} file(s) in '{args.input_dir}'.")
        print(f"Output directory: '{args.output_dir}'")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = []
        batch_start = time.time()
        
        for filename in input_files:
            file_path = os.path.join(args.input_dir, filename)
            name_without_ext = os.path.splitext(filename)[0].strip()
            
            # Create a sub-directory for each file to avoid collisions
            file_output_dir = os.path.join(args.output_dir, name_without_ext)
            output_path = os.path.join(file_output_dir, f"{name_without_ext}.wav")
            
            if is_rvc:
                result = process_single_audio_rvc(file_path, output_path, args)
            else:
                result = process_single_file(file_path, output_path, engine, args)
            results.append(result)
            
        batch_elapsed = time.time() - batch_start
        
        print_summary_table(results)
        print(f"\nTotal batch time: {batch_elapsed:.1f}s")


if __name__ == "__main__":
    main()
