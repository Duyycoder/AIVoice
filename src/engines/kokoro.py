import os
import sys
import torch
import soundfile as sf
from src.engines.base import BaseTTSEngine

class KokoroEngine(BaseTTSEngine):
    """Adapter for Kokoro-Vietnamese engine running PyTorch model."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.tts = None
        self.current_voice = None
        self._active_gpu = None
        
        # Lazy imports check
        try:
            import kokoro_vietnamese
        except ImportError:
            pass

    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        # Directory Safeguard
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        try:
            # Resolve parameters
            voice = kwargs.get("voice") or "diem_trinh"
            valid_kokoro_voices = [
                "diem_trinh", "hung_thinh", "mai_linh", "mai_loan", "manh_dung", 
                "my_yen", "ngoc_huyen", "phat_tai", "thanh_dat", "thuc_trinh", 
                "tuan_ngoc", "storyvert", "duc_an", "duc_duy"
            ]
            if voice not in valid_kokoro_voices:
                voice = "diem_trinh"
                
            device_opt = kwargs.get("device") or "cuda"
            gpu = torch.cuda.is_available() and (device_opt != "cpu")
            
            if not hasattr(self, "_active_gpu"):
                self._active_gpu = None
                
            # State caching checks to avoid redundant reload
            if not self.tts or gpu != self._active_gpu or voice != self.current_voice:
                print(f"Initializing Kokoro-Vietnamese (device={'cuda' if gpu else 'cpu'}, voice={voice})...")
                from kokoro_vietnamese import KokoroVietnamese
                self.tts = KokoroVietnamese(device="cuda" if gpu else "cpu", voice=voice)
                self._active_gpu = gpu
                self.current_voice = voice
                
            # Synthesize
            audio, phonemes = self.tts.synthesize(text)
            
            # Save audio (Kokoro-Vietnamese outputs mono at 24000Hz)
            sf.write(output_path, audio, 24000)
            
            # Verify file exists and has size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            return False
            
        except Exception as e:
            print(f"KokoroEngine generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
