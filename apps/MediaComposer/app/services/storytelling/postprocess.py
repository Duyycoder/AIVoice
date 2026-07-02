import os
from typing import Callable, List, Optional
from PIL import Image, ImageOps
import numpy as np
from loguru import logger

try:
    import torch
except ImportError:
    pass

from app.services.storytelling.image_generator import StorytellingPipeline
from app.services.storytelling.face_extractor import _get_face_app
from app.config import load_storytelling_config

# Absolute path to MediaComposer root (safe regardless of CWD)
_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class PostProcessor:
    def __init__(self, device: str = "cuda", enable_upscaling: Optional[bool] = None):
        self.device = device
        self._realesrgan_model = None
        # enable_upscaling: None means read from config at render time (default behaviour)
        self._enable_upscaling_override: Optional[bool] = enable_upscaling

    def run_adetailer(
        self,
        image: Image.Image,
        pipeline: StorytellingPipeline,
        detection_threshold: float = 0.5
    ) -> Image.Image:
        
        try:
            import cv2
            app = _get_face_app(self.device)
            open_cv_image = np.array(image) 
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            faces = app.get(open_cv_image)
        except Exception as e:
            logger.warning(f"ADetailer face detection failed: {e}")
            return image
            
        if not faces:
            return image
            
        logger.info(f"ADetailer: Detected {len(faces)} faces. Inpainting is simulated in this MVP.")
        return image

    def run_realesrgan(
        self,
        image: Image.Image,
        scale: int = 4,
        model_name: Optional[str] = None,
        target_w: Optional[int] = None,
        target_h: Optional[int] = None
    ) -> Image.Image:
        
        st_config = load_storytelling_config()
        if model_name is None:
            model_name = st_config.get("upscaler_model", "RealESRGAN_x4plus_anime_6B")
        if target_w is None:
            target_w = st_config.get("output_width", 1920)
        if target_h is None:
            target_h = st_config.get("output_height", 1080)

        # Determine enable_upscaling: instance override (from orchestrator) takes priority over config
        enable_upscaling = self._enable_upscaling_override
        if enable_upscaling is None:
            enable_upscaling = st_config.get("enable_upscaling", True)

        if not enable_upscaling:
            return ImageOps.fit(image, (target_w, target_h), Image.Resampling.LANCZOS)

        # Use absolute path to avoid CWD dependency
        weight_path = os.path.join(_MC_ROOT, "models", "realesrgan", f"{model_name}.pth")
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            import cv2

            if not os.path.exists(weight_path):
                logger.warning(f"RealESRGAN weights not found at {weight_path}. Using PIL resize instead.")
                w, h = image.size
                upscaled = image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
            else:
                if self._realesrgan_model is None:
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                    self._realesrgan_model = RealESRGANer(
                        scale=scale,
                        model_path=weight_path,
                        model=model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=True if self.device == "cuda" else False,
                        device=torch.device(self.device)
                    )

                img_cv = np.array(image)[:, :, ::-1]
                output, _ = self._realesrgan_model.enhance(img_cv, outscale=scale)
                upscaled = Image.fromarray(output[:, :, ::-1])

        except Exception as e:
            logger.warning(f"RealESRGAN failed: {e}. Using PIL resize.")
            w, h = image.size
            upscaled = image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

        return ImageOps.fit(upscaled, (target_w, target_h), Image.Resampling.LANCZOS)


    def process_all(
        self,
        frame_paths: List[str],
        output_dir: str,
        pipeline: StorytellingPipeline,
        on_progress: Callable[[int, int], None] = None
    ) -> List[str]:
        
        os.makedirs(output_dir, exist_ok=True)
        final_paths = []
        total = len(frame_paths)
        
        for i, path in enumerate(frame_paths):
            if not os.path.exists(path):
                logger.warning(f"Frame not found: {path}")
                continue
                
            img = Image.open(path).convert("RGB")
            
            img = self.run_adetailer(img, pipeline)
            
            img = self.run_realesrgan(img)
            
            filename = os.path.basename(path)
            out_path = os.path.join(output_dir, filename)
            img.save(out_path, format="PNG")
            final_paths.append(out_path)
            
            if on_progress:
                on_progress(i + 1, total)
                
        return final_paths
