import os
import torch
from src.engines.base import BaseTTSEngine

class CloneEngine(BaseTTSEngine):
    """Adapter for XTTSv2 zero-shot voice cloning engine running local models."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.tts = None
        self.current_ref_audio = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        
        # Verify TTS package can be imported, so main.py's try-except block can catch ImportError if missing.
        # This keeps imports lazy at module level but provides prompt error handling.
        try:
            import TTS
        except ImportError:
            raise ImportError("coqui-tts is not installed. Please run `pip install TTS`.")

    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        # Check model path override, initialization path, or default path
        model_p = kwargs.get("model") or self.model_path or os.path.join("models", "xtts_v2")
        
        if not os.path.exists(model_p):
            raise FileNotFoundError(f"XTTSv2 model directory not found at: {model_p}")
            
        # Check device override ("cpu" vs "cuda")
        device_opt = kwargs.get("device") or "cuda"
        gpu = torch.cuda.is_available() and (device_opt != "cpu")
        
        # Track device change to trigger model reload if necessary
        if not hasattr(self, "_active_gpu"):
            self._active_gpu = None
            
        if not self.tts or model_p != self.model_path or gpu != self._active_gpu:
            # Monkeypatch torchaudio.load and torchaudio.save to bypass torchcodec/FFmpeg DLL issues
            import torchaudio
            import soundfile as sf
            
            def patched_load(filepath, *args, **kwargs):
                data, samplerate = sf.read(filepath, dtype='float32', always_2d=True)
                tensor = torch.from_numpy(data).t()
                return tensor, samplerate
                
            def patched_save(filepath, src, sample_rate, channels_first=True, *args, **kwargs):
                data = src.cpu().numpy()
                if channels_first and len(data.shape) > 1:
                    data = data.T
                sf.write(filepath, data, sample_rate)
                
            torchaudio.load = patched_load
            torchaudio.save = patched_save
            
            # Monkeypatch VoiceBpeTokenizer.preprocess_text to support 'vi' language preprocessing
            try:
                from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
                orig_preprocess_text = VoiceBpeTokenizer.preprocess_text
                
                def patched_preprocess_text(self, txt, lang):
                    if lang == "vi":
                        from TTS.tts.utils.text.cleaners import basic_cleaners
                        return basic_cleaners(txt)
                    return orig_preprocess_text(self, txt, lang)
                    
                VoiceBpeTokenizer.preprocess_text = patched_preprocess_text
            except Exception as e:
                print(f"Warning: Failed to monkeypatch VoiceBpeTokenizer: {e}")
                
            # Lazy loading of TTS class and loading the model weights
            from TTS.api import TTS
            
            print(f"Loading local XTTSv2 model from: {model_p} (Using GPU: {gpu})")
            
            # Load the model locally. Pass the folder path containing model.pth and other files as model_path,
            # and the config.json file path as config_path.
            config_file = os.path.join(model_p, "config.json")
            
            # Temporary monkeypatch of torch.load to bypass weights_only check in PyTorch 2.6+
            orig_load = torch.load
            try:
                # Wrap torch.load to force weights_only=False
                torch.load = lambda *a, **kw: orig_load(*a, **{**kw, 'weights_only': False})
                self.tts = TTS(model_path=model_p, config_path=config_file, gpu=gpu)
            finally:
                torch.load = orig_load
                
            self.model_path = model_p
            self._active_gpu = gpu
            
        ref_audio = kwargs.get("ref_audio")
        if not ref_audio:
            raise ValueError("Reference audio path (--ref_audio) is required for CloneEngine.")
            
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio file not found at: {ref_audio}")
            
        # Map the CLI --voice parameter to the target language code, defaulting to 'en'
        lang = kwargs.get("voice") or "en"
        
        # Check if the language is supported by XTTSv2, otherwise fallback to 'en'
        supported_langs = getattr(self.tts, "languages", [])
        if supported_langs and lang not in supported_langs:
            print(f"Warning: Language '{lang}' is not supported by XTTSv2. Supported languages: {supported_langs}. Falling back to 'en'.")
            lang = "en"
            
        speed = kwargs.get("speed", 1.0)
        use_fp16 = kwargs.get("use_fp16", True) and gpu
        use_tf32 = kwargs.get("use_tf32", True) and gpu
        
        # Apply local TF32 settings
        if gpu:
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32
            
        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        try:
            import contextlib
            
            # Use torch's native SDPA context manager for efficient GPU inference if GPU is available
            if gpu:
                sdpa_context = torch.backends.cuda.sdp_kernel(
                    enable_flash=True, 
                    enable_math=True, 
                    enable_mem_efficient=True
                )
            else:
                sdpa_context = contextlib.nullcontext()
                
            with sdpa_context:
                abs_ref = os.path.abspath(ref_audio)
                model = self.tts.synthesizer.tts_model
                
                # Cache speaker latents/embeddings to speed up inference significantly
                if (self.current_ref_audio != abs_ref or 
                    self.gpt_cond_latent is None or 
                    self.speaker_embedding is None):
                    
                    print(f"Computing speaker latents for reference audio: {ref_audio}")
                    # Always compute latents in standard float32 precision
                    model.float()
                    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=abs_ref)
                    self.gpt_cond_latent = gpt_cond_latent
                    self.speaker_embedding = speaker_embedding
                    self.current_ref_audio = abs_ref
                
                run_fp16 = use_fp16
                try:
                    if run_fp16:
                        # Use dynamic PyTorch mixed precision (autocast) which is stable and avoids mismatch errors
                        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                    else:
                        autocast_ctx = contextlib.nullcontext()
                        
                    # Model and inputs remain in Float32, autocast handles downcasting internally
                    model.float()
                    gpt_cond = self.gpt_cond_latent.float()
                    spk_emb = self.speaker_embedding.float()
                    
                    with autocast_ctx:
                        # Perform fast inference
                        out = model.inference(
                            text=text,
                            language=lang,
                            gpt_cond_latent=gpt_cond,
                            speaker_embedding=spk_emb,
                            speed=speed,
                            temperature=0.65,
                            repetition_penalty=2.0
                        )
                    
                    wav = out['wav']
                    if isinstance(wav, torch.Tensor):
                        wav_cpu = wav.cpu().numpy()
                    else:
                        wav_cpu = wav
                        
                    # NaN and Inf checking
                    import numpy as np
                    has_nan = np.isnan(wav_cpu).any() or np.isinf(wav_cpu).any()
                    
                    if has_nan and run_fp16:
                        print("Warning: NaN or Inf detected in FP16 output! Reverting to FP32 in-memory...")
                        raise ValueError("FP16 NaN/Inf output detected.")
                        
                except Exception as e:
                    if run_fp16:
                        print(f"FP16 autocast failed/NaN ({e}). Retrying chunk in FP32 fallback...")
                        
                        out = model.inference(
                            text=text,
                            language=lang,
                            gpt_cond_latent=self.gpt_cond_latent.float(),
                            speaker_embedding=self.speaker_embedding.float(),
                            speed=speed,
                            temperature=0.65,
                            repetition_penalty=2.0
                        )
                        wav = out['wav']
                    else:
                        raise e
                
                import soundfile as sf
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                # soundfile does not support float16, cast to float32 if needed
                import numpy as np
                if wav.dtype == np.float16:
                    wav = wav.astype(np.float32)
                sf.write(output_path, wav, 24000) # XTTSv2 sample rate is 24000Hz
                
                # Aggressive VRAM cleaning for RTX 3060 and CPU profiles
                profile_opt = kwargs.get("hardware_profile") or "rtx3060"
                if not gpu or profile_opt == "rtx3060":
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("Aggressively cleared CUDA VRAM after chunk processing.")
                
            return True
        except Exception as e:
            print(f"CloneEngine voice cloning failed: {e}")
            return False
