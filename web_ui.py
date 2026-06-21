import os
import sys
import gc
import json
import time
import glob
import asyncio
import threading
import contextlib
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory

# Configure system search path for DLLs if on Windows
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

# Force TF32 defaults early
try:
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    pass

app = Flask(__name__, template_folder='templates')

# Global Lock to serialize GPU inference requests
gpu_inference_lock = threading.Lock()

# Lock log list to capture and report logs to the frontend
logs_buffer = []
logs_lock = threading.Lock()

def add_log(message):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with logs_lock:
        logs_buffer.append(formatted)

def clear_logs():
    with logs_lock:
        logs_buffer.clear()

# Paths to block for security (System folders & internal code folders)
BLOCKED_KEYWORDS = [
    "c:\\windows", "system32", "program files", "system volume information",
    "\\src\\", "\\models\\", "\\.venv\\", "\\.git\\", "\\.vscode\\",
    "\\configs\\", "\\tests\\"
]

def sanitize_and_validate_path(path_str: str, is_output: bool = False) -> str:
    """Validates absolute and relative paths, preventing path traversal

    and accidental overwrites of system or internal project directories.
    """
    if not path_str:
        return ""
        
    normalized = os.path.normpath(path_str)
    lower_path = normalized.lower()
    
    # 1. Block known critical system and project directories
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lower_path:
            raise ValueError(f"Đường dẫn chứa thư mục bị chặn vì lý do bảo mật: '{keyword}'")
            
    # 2. Prevent writing executable or config files
    if is_output:
        base, ext = os.path.splitext(lower_path)
        if ext in ['.py', '.json', '.bat', '.sh', '.exe', '.dll', '.cmd']:
            raise ValueError("Không được phép ghi đè các tệp thực thi hoặc cấu hình (.py, .json, .bat, v.v.)")
            
    # 3. For input files/directories, verify existence
    if not is_output and not os.path.exists(normalized):
        raise ValueError(f"Đường dẫn đầu vào không tồn tại: {normalized}")
        
    # 4. Verify write permission on output directories
    if is_output:
        out_dir = normalized if os.path.isdir(normalized) else os.path.dirname(normalized)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            # Try writing a temporary file to test permissions
            try:
                test_file = os.path.join(out_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                raise PermissionError(f"Không có quyền ghi vào thư mục đầu ra: {out_dir}. Lỗi: {e}")
                
    return os.path.abspath(normalized)

def open_file_dialog_subprocess() -> str:
    """Launches a separate Python process to run a native Tkinter file selection dialog.

    This prevents thread blockages or Tkinter mainloop focus crashes in Flask.
    """
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        "import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); root.attributes('-topmost', True); print(filedialog.askopenfilename(title='Chọn tệp tin đầu vào (.md, .txt)', filetypes=[('Văn bản / Markdown', '*.md;*.txt')])); root.destroy()"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return res.stdout.strip()
    except Exception as e:
        print(f"Error in file dialog subprocess: {e}")
        return ""

def open_dir_dialog_subprocess() -> str:
    """Launches a separate Python process to run a native Tkinter folder selection dialog."""
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        "import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); root.attributes('-topmost', True); print(filedialog.askdirectory(title='Chọn thư mục')); root.destroy()"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return res.stdout.strip()
    except Exception as e:
        print(f"Error in dir dialog subprocess: {e}")
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/picker/file')
def pick_file():
    path = open_file_dialog_subprocess()
    return jsonify({"path": path})

@app.route('/api/picker/directory')
def pick_directory():
    path = open_dir_dialog_subprocess()
    return jsonify({"path": path})

@app.route('/api/logs')
def get_logs():
    with logs_lock:
        return jsonify({"logs": list(logs_buffer)})

@app.route('/api/voices')
def list_voices():
    """Dynamically fetches Vietnamese voices available in edge-tts."""
    try:
        import edge_tts
        async def fetch():
            voices = await edge_tts.VoicesManager.create()
            vi_voices = voices.find(Language="vi")
            return [{"ShortName": v["ShortName"], "Gender": v["Gender"]} for v in vi_voices]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        vi_voices = loop.run_until_complete(fetch())
        loop.close()
        return jsonify({"voices": vi_voices})
    except Exception as e:
        # Fallback list if edge_tts query fails
        return jsonify({"voices": [
            {"ShortName": "vi-VN-NamMinhNeural", "Gender": "Male"},
            {"ShortName": "vi-VN-HoaiMyNeural", "Gender": "Female"}
        ]})

@app.route('/api/models')
def list_models():
    """Lists locally downloaded models in models/ and voices/."""
    models_dir = "models"
    voices_dir = os.path.join("data", "voices")
    
    piper_models = [os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "piper", "*.onnx"))]
    
    xtts_models = []
    if os.path.exists(os.path.join(models_dir, "xtts_v2")):
        xtts_models = [os.path.basename(d) for d in os.listdir(os.path.join(models_dir, "xtts_v2")) if os.path.isdir(os.path.join(models_dir, "xtts_v2", d))]
        if os.path.exists(os.path.join(models_dir, "xtts_v2", "model.pth")):
            xtts_models.append("xtts_v2 (Default)")
         
    rvc_models = [os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "rvc", "*.pth"))]
    rvc_indexes = [os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "rvc", "*.index"))]
    llm_models = [os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "llm", "*.gguf"))]
    
    ref_audios = [os.path.basename(p) for p in glob.glob(os.path.join(voices_dir, "*.wav"))]
    
    # Query GPU properties
    vram_gb = 0.0
    gpu_name = ""
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            vram_gb = round(prop.total_memory / (1024 ** 3), 2)
            gpu_name = prop.name
    except Exception:
        pass
    
    return jsonify({
        "piper": piper_models,
        "xtts": xtts_models,
        "rvc": rvc_models,
        "rvc_index": rvc_indexes,
        "llm": llm_models,
        "ref_audio": ref_audios,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb
    })

@app.route('/api/diagnose', methods=['GET'])
def diagnose_gpu():
    """Executes check_gpu.py and returns the console report."""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "check_gpu.py"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return jsonify({"report": result.stdout})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_speech():
    clear_logs()
    data = request.json or {}
    
    direct_text = data.get("direct_text", "").strip()
    temp_file_to_clean = None
    
    # 1. Parse and sanitize input/output paths
    input_path_raw = data.get("input_path", "")
    if direct_text:
        import tempfile
        fd, temp_file_path = tempfile.mkstemp(suffix="_direct_text.txt", prefix="aivoice_")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(direct_text)
        input_path_raw = temp_file_path
        temp_file_to_clean = temp_file_path
        
    input_dir_raw = data.get("input_dir", "")
    output_dir_raw = data.get("output_dir", "data/outputs")
    output_name = data.get("output_name", "")
    
    is_batch = bool(input_dir_raw)
    
    try:
        if is_batch:
            input_dir = sanitize_and_validate_path(input_dir_raw, is_output=False)
            input_path = None
        else:
            input_path = sanitize_and_validate_path(input_path_raw, is_output=False)
            input_dir = None
            
        output_dir = sanitize_and_validate_path(output_dir_raw, is_output=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
        
    # 2. Gather parameters
    engine_name = data.get("engine", "edge")
    model_name = data.get("model", "")
    voice = data.get("voice", "")
    speed = float(data.get("speed", 1.0))
    ref_audio = data.get("ref_audio", "")
    phonemize = bool(data.get("phonemize", False))
    normalize = bool(data.get("normalize", True))
    target_lufs = float(data.get("target_lufs", -14.0))
    fade_in = float(data.get("fade_in", 0.1))
    fade_out = float(data.get("fade_out", 0.1))
    silence_duration = float(data.get("silence_duration", 0.3))
    
    # Local GGUF LLM Spicing
    spice_text = bool(data.get("spice_text", False))
    llm_model = data.get("llm_model", "")
    
    # RVC
    rvc_model = data.get("rvc_model", "")
    rvc_index = data.get("rvc_index", "")
    rvc_pitch = int(data.get("rvc_pitch", 0))
    
    # Hardware profile options: 'rtx5060', 'rtx3060', or 'cpu'
    hardware_profile = data.get("hardware_profile", "rtx3060")
    
    if hardware_profile == "rtx5060":
        device = "cuda"
        use_fp16 = True
        use_tf32 = True
        max_words = 50
    elif hardware_profile == "rtx3060":
        device = "cuda"
        use_fp16 = True
        use_tf32 = True
        max_words = 30
    else: # cpu
        device = "cpu"
        use_fp16 = False
        use_tf32 = False
        max_words = 30
    
    # Resolve relative helper paths
    if model_name:
        if engine_name == "piper":
            model_name = os.path.abspath(os.path.join("models", "piper", model_name))
        elif engine_name == "clone":
            model_name = os.path.abspath(os.path.join("models", "xtts_v2"))
            
    if ref_audio:
        ref_audio = os.path.abspath(os.path.join("data", "voices", ref_audio))
    if llm_model:
        llm_model = os.path.abspath(os.path.join("models", "llm", llm_model))
    if rvc_model:
        rvc_model = os.path.abspath(os.path.join("models", "rvc", rvc_model))
    if rvc_index:
        rvc_index = os.path.abspath(os.path.join("models", "rvc", rvc_index))

    # 3. Create dummy Namespace arg object to feed into the pipeline
    import argparse
    args = argparse.Namespace()
    args.input = input_path
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.output_name = output_name or None
    args.engine = engine_name
    args.model = model_name
    args.voice = voice
    args.speed = speed
    args.ref_audio = ref_audio
    args.phonemize = phonemize
    args.normalize = normalize
    args.target_lufs = target_lufs
    args.fade_in = fade_in
    args.fade_out = fade_out
    args.silence_duration = silence_duration
    args.spice_text = spice_text
    args.llm_model = llm_model
    args.rvc_model = rvc_model
    args.rvc_index = rvc_index
    args.rvc_pitch = rvc_pitch
    args.device = device
    args.use_fp16 = use_fp16
    args.use_tf32 = use_tf32
    args.max_words = max_words
    args.hardware_profile = hardware_profile
    args.is_direct_text = bool(direct_text)
    
    # Define generation wrapper to be run inside the serialized lock
    def run_generation():
        is_gpu_task = (device == "cuda" and engine_name in ["clone", "rvc"]) or bool(rvc_model) or spice_text
        
        if is_gpu_task and gpu_inference_lock.locked():
            add_log("Yêu cầu đang được xếp hàng chờ xử lý (GPU đang bận)...")
            
        # Serialize GPU tasks
        with gpu_inference_lock if is_gpu_task else contextlib.nullcontext():
            add_log(f"Bắt đầu xử lý TTS bằng động cơ: {engine_name.upper()}...")
            add_log(f"Cấu hình GPU: {hardware_profile.upper()} | Thiết bị: {device.upper()} | FP16: {use_fp16} | TF32: {use_tf32} | Độ dài đoạn: {max_words} từ")
            
            # Setup TF32 globally inside the worker thread
            try:
                import torch
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = use_tf32
                    torch.backends.cudnn.allow_tf32 = use_tf32
            except ImportError:
                pass
                
            # Initialize engine plugin
            engine = None
            if engine_name == "piper":
                from src.engines.piper import PiperEngine
                engine = PiperEngine(args.model)
            elif engine_name == "edge":
                from src.engines.edge import EdgeEngine
                engine = EdgeEngine(args.voice)
            elif engine_name == "clone":
                from src.engines.clone import CloneEngine
                engine = CloneEngine(args.model)
                
            from main import process_single_file, process_single_audio_rvc
            
            if args.input:
                if getattr(args, "is_direct_text", False):
                    out_name = (args.output_name or f"direct_text_{int(time.time())}").strip()
                    out_path = os.path.join(args.output_dir, "direct_text", f"{out_name}.wav")
                else:
                    input_base_name = os.path.splitext(os.path.basename(args.input))[0].strip()
                    out_name = (args.output_name or input_base_name).strip()
                    out_path = os.path.join(args.output_dir, input_base_name, f"{out_name}.wav")
                
                if args.engine == "rvc":
                    result = process_single_audio_rvc(args.input, out_path, args)
                else:
                    result = process_single_file(args.input, out_path, engine, args)
                    
                if result["status"] == "SUCCESS":
                    add_log(f"Hoàn thành thành công! File lưu tại: {result['output']}")
                    return True, [result]
                else:
                    add_log(f"Lỗi xử lý: {result['status']}")
                    return False, [result]
            else:
                # Batch processing
                import glob
                files = glob.glob(os.path.join(args.input_dir, "*.md")) + glob.glob(os.path.join(args.input_dir, "*.txt"))
                if not files:
                    add_log("Không tìm thấy tệp tin .md hoặc .txt nào trong thư mục đầu vào.")
                    return False, []
                    
                add_log(f"Tìm thấy {len(files)} tệp tin cần xử lý...")
                results = []
                all_success = True
                
                for idx, file_p in enumerate(files):
                    add_log(f"[{idx+1}/{len(files)}] Đang xử lý tệp: {os.path.basename(file_p)}...")
                    input_base_name = os.path.splitext(os.path.basename(file_p))[0].strip()
                    out_path = os.path.join(args.output_dir, input_base_name, f"{input_base_name}.wav")
                    
                    res = process_single_file(file_p, out_path, engine, args)
                    results.append(res)
                    if res["status"] != "SUCCESS":
                        all_success = False
                        
                add_log("Hoàn thành xử lý hàng loạt.")
                return all_success, results

    # Run in a background thread to prevent blocking Flask response
    def async_wrapper():
        try:
            success, results = run_generation()
        except Exception as e:
            import traceback
            traceback.print_exc()
            add_log(f"LỖI HỆ THỐNG: {e}")
        finally:
            if temp_file_to_clean and os.path.exists(temp_file_to_clean):
                try:
                    os.remove(temp_file_to_clean)
                except OSError:
                    pass
            
    threading.Thread(target=async_wrapper).start()
    return jsonify({"success": True, "message": "Quá trình sinh âm thanh đã bắt đầu."})

@app.route('/api/audio')
def get_audio_file():
    """Serves generated audio files given an absolute path."""
    path = request.args.get('path', '')
    if not path:
        return jsonify({"error": "Thiếu đường dẫn tệp"}), 400
        
    try:
        abs_path = sanitize_and_validate_path(path, is_output=False)
        if not abs_path.endswith('.wav'):
            return jsonify({"error": "Định dạng tệp không hợp lệ"}), 400
        return send_file(abs_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/history_list')
def get_history():
    """Returns a list of generated audio files in the output directory."""
    output_dir = request.args.get('path', 'data/outputs')
    try:
        abs_dir = sanitize_and_validate_path(output_dir, is_output=False)
        # Search recursively for wav files
        wav_files = glob.glob(os.path.join(abs_dir, "**", "*.wav"), recursive=True)
        # Sort by creation time (newest first)
        wav_files.sort(key=os.path.getmtime, reverse=True)
        
        history = []
        for p in wav_files[:20]: # Limit to top 20
            history.append({
                "name": os.path.basename(p),
                "path": p,
                "created": time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(os.path.getmtime(p))),
                "size": f"{os.path.getsize(p) / (1024*1024):.2f} MB"
            })
        return jsonify({"history": history})
    except Exception:
        return jsonify({"history": []})

if __name__ == '__main__':
    # Listen strictly to localhost 127.0.0.1 for maximum security
    app.run(host='127.0.0.1', port=5000, debug=False)
