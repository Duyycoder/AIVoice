import os
import sys
from src.engines.base import BaseTTSEngine

class ValtecEngine(BaseTTSEngine):
    """Adapter for Valtec-TTS engine."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.tts = None
        
        # Verify it can be imported, keeping imports lazy but providing error handling
        try:
            import valtec_tts
        except ImportError:
            pass

    def _load_model(self):
        if self.tts is None:
            print("Loading Valtec-TTS model...")
            import sys
            import valtec_tts
            
            # 1. Backup all modules starting with 'src' or 'src.' from sys.modules
            src_keys = [k for k in sys.modules.keys() if k == 'src' or k.startswith('src.')]
            src_backup = {k: sys.modules[k] for k in src_keys}
            for k in src_keys:
                sys.modules.pop(k, None)
                
            # 2. Insert valtec-tts repository root path to sys.path at index 0
            valtec_dir = os.path.dirname(valtec_tts.__file__)
            valtec_repo_root = os.path.dirname(valtec_dir)
            if valtec_repo_root not in sys.path:
                sys.path.insert(0, valtec_repo_root)
                
            try:
                # 3. Import and load VITS TTS model
                from valtec_tts import TTS
                self.tts = TTS()
            finally:
                # 4. Clean up Valtec's 'src' submodules and restore AIVoice's 'src' submodules
                valtec_src_keys = [k for k in sys.modules.keys() if k == 'src' or k.startswith('src.')]
                for k in valtec_src_keys:
                    sys.modules.pop(k, None)
                sys.modules.update(src_backup)

    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        # Directory Safeguard
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        try:
            # Lazy loading
            self._load_model()
            
            # Extract speaker (default to NF)
            # Valtec speaker profiles: NF, SF, NM1, NM2, SM
            speaker = kwargs.get("voice") or kwargs.get("speaker") or "NF"
            valid_speakers = ["NF", "SF", "NM1", "NM2", "SM"]
            if speaker not in valid_speakers:
                speaker = "NF"
            
            # Valtec speak function
            self.tts.speak(text, speaker=speaker, output_path=output_path)
            
            # Verify file exists and has size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            return False
            
        except Exception as e:
            print(f"ValtecEngine generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
