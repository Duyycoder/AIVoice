import os
import sys
from src.engines.base import BaseTTSEngine

class VieNeuEngine(BaseTTSEngine):
    """Adapter for VieNeu-TTS engine."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.tts = None
        self.current_mode = None
        self.current_codec_repo = None
        
        # Lazy imports check
        try:
            import vieneu
        except ImportError:
            pass

    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        # Directory Safeguard
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        try:
            # Resolve parameters
            mode = kwargs.get("vieneu_mode") or "v3turbo"
            ref_audio = kwargs.get("ref_audio")
            ref_text = kwargs.get("ref_text")
            emotion = kwargs.get("vieneu_emotion") or "natural"
            temp_val = kwargs.get("temperature")
            temperature = float(temp_val) if temp_val is not None else 0.8
            
            # Determine appropriate codec repository (voice clone needs PyTorch encoder/decoder)
            voice = kwargs.get("voice")
            is_cloning = (voice == "ref_audio" and ref_audio is not None)
            
            codec_repo = None
            if mode == "standard":
                codec_repo = "neuphonic/neucodec" if is_cloning else "neuphonic/neucodec-onnx-decoder-int8"
            
            # State caching checks to avoid redundant reload
            if (not self.tts or 
                mode != self.current_mode or 
                (mode == "standard" and codec_repo != getattr(self, "current_codec_repo", None))):
                
                print(f"Initializing VieNeu-TTS (mode={mode}, codec_repo={codec_repo})...")
                from vieneu import Vieneu
                init_kwargs = {"mode": mode}
                if codec_repo:
                    init_kwargs["codec_repo"] = codec_repo
                self.tts = Vieneu(**init_kwargs)
                self.current_mode = mode
                self.current_codec_repo = codec_repo
                
            # Synthesize
            infer_kwargs = {
                "text": text,
                "emotion": emotion,
                "temperature": temperature,
                "max_new_frames": 800
            }
            
            voice = kwargs.get("voice")
            if voice == "ref_audio" and ref_audio:
                infer_kwargs["ref_audio"] = ref_audio
                if ref_text:
                    infer_kwargs["ref_text"] = ref_text
            elif voice and voice != "ref_audio" and voice != "vi-VN-NamMinhNeural":
                infer_kwargs["voice"] = voice
                
            audio = self.tts.infer(**infer_kwargs)
                
            # Save audio using the library's save helper
            self.tts.save(audio, output_path)
            
            # Verify file exists and has size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            return False
            
        except Exception as e:
            print(f"VieNeuEngine generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
