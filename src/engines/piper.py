import os
import wave
import subprocess
import sys
import unicodedata
from src.engines.base import BaseTTSEngine

class PiperEngine(BaseTTSEngine):
    """Adapter for the Piper TTS engine running local ONNX models."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.voice = None
        
        # Lazy loading on first generation to keep initialization lightweight
        if model_path and os.path.exists(model_path):
            try:
                self._load_model(model_path)
            except (ImportError, ModuleNotFoundError):
                pass
            
    def _load_model(self, path: str):
        from piper.voice import PiperVoice
        import onnxruntime
        use_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        self.voice = PiperVoice.load(path, use_cuda=use_cuda)
        self.model_path = path

    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        # Check model path override or default path
        model_p = kwargs.get("model") or self.model_path
        if not model_p:
            raise ValueError("Piper model path must be specified via --model or initialization.")
            
        if not os.path.exists(model_p):
            raise FileNotFoundError(f"Piper ONNX model file not found at: {model_p}")
            
        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        # 1. Ép chuẩn tiếng Việt NFC (Gộp chữ và dấu) trước khi truyền vào model
        text = unicodedata.normalize('NFC', text)
        
        # Task 2: Force UTF-8 Subprocess Encoding
        # Get speed mapping: length_scale is inversely proportional to speed
        speed = kwargs.get("speed", 1.0)
        length_scale = 1.0 / speed if speed > 0 else 1.0
        
        # Try python library (in-process) generation first. It's much faster,
        # avoids launching hundreds of subprocesses, and prevents memory/OS resource exhaustion.
        try:
            if not self.voice or model_p != self.model_path:
                self._load_model(model_p)
                
            from piper import SynthesisConfig
            speaker_id = None
            voice_val = kwargs.get("voice")
            if voice_val is not None:
                try:
                    speaker_id = int(voice_val)
                except ValueError:
                    pass
            syn_config = SynthesisConfig(length_scale=length_scale, speaker_id=speaker_id)
            
            with wave.open(output_path, "wb") as wav_file:
                # Explicitly set WAV parameters (Piper is mono, 16-bit)
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.voice.config.sample_rate)
                
                self.voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=syn_config,
                    set_wav_format=False
                )
            return True
            
        except Exception as e_lib:
            print(f"Piper library generation failed: {e_lib}. Falling back to subprocess execution.")
            
            # Determine path to piper executable. Under .venv/Scripts/piper.exe
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            proj_dir = os.path.dirname(src_dir)
            piper_exe = os.path.join(proj_dir, ".venv", "Scripts", "piper.exe")
            if not os.path.exists(piper_exe):
                venv_bin = os.path.dirname(sys.executable)
                piper_exe = os.path.join(venv_bin, "piper.exe")
                if not os.path.exists(piper_exe):
                    piper_exe = "piper"  # Fallback to system PATH
                
            # 2. Ép môi trường Windows sử dụng UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            cmd = [
                piper_exe,
                "--model", model_p,
                "--output_file", output_path,
                "--length_scale", str(length_scale)
            ]
            voice_val = kwargs.get("voice")
            if voice_val is not None:
                cmd.extend(["--speaker", str(voice_val)])
                
            import onnxruntime
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                cmd.append("--cuda")
            
            try:
                # 3. Cập nhật đoạn gọi subprocess
                # Đảm bảo bạn chèn thêm tham số encoding='utf-8' và env=env vào hàm
                process = subprocess.run(
                    cmd, # Danh sách lệnh piper.exe của bạn vẫn giữ nguyên
                    input=text,
                    text=True,
                    encoding='utf-8',
                    env=env,
                    capture_output=True,
                    check=True
                )
                return True
            except Exception as e_sub:
                print(f"Piper subprocess execution failed: {e_sub}")
                return False
