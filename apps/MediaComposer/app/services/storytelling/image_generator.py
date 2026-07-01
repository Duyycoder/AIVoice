import os
from typing import Optional, List, Tuple
from loguru import logger
from PIL import Image
import numpy as np

try:
    import torch
    # --- HOTFIX FOR PYTORCH 2.6.0 + TORCHAO ---
    import torch.utils._pytree
    if not hasattr(torch.utils._pytree, 'register_constant'):
        torch.utils._pytree.register_constant = lambda *args, **kwargs: None
    # ------------------------------------------
    from diffusers import StableDiffusionPipeline, LCMScheduler, AutoencoderTiny
except ImportError:
    pass

from app.services.storytelling.models import StoryContext

class StyleBuffer:
    MAX_SIZE = 8
    WEIGHT = 0.25
    
    def __init__(self):
        self.embeddings: List['torch.Tensor'] = []
        
    def add_accepted_image(self, image: Image.Image, pipe: 'StableDiffusionPipeline' = None) -> None:
        """
        Dùng IP-Adapter image encoder để tính embedding của ảnh.
        """
        if pipe is None or not hasattr(pipe, "image_encoder") or pipe.image_encoder is None:
            logger.warning("Pipeline doesn't have image_encoder, cannot add to style buffer")
            return
            
        try:
            import torchvision.transforms as T
            if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
                clip_image = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
            else:
                transform = T.Compose([
                    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                ])
                clip_image = transform(image).unsqueeze(0)
                
            clip_image = clip_image.to(pipe.device, dtype=pipe.image_encoder.dtype)
            
            with torch.no_grad():
                image_embeds = pipe.image_encoder(clip_image).image_embeds
                
            self.embeddings.append(image_embeds)
            if len(self.embeddings) > self.MAX_SIZE:
                self.embeddings.pop(0)
        except Exception as e:
            logger.error(f"Failed to extract style embedding: {e}")

    def get_style_embedding(self) -> Optional['torch.Tensor']:
        if not self.embeddings:
            return None
        import torch
        return torch.mean(torch.cat(self.embeddings, dim=0), dim=0, keepdim=True)

    def clear(self) -> None:
        self.embeddings.clear()


class StorytellingPipeline:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(StorytellingPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, context: StoryContext = None, device: str = "cuda"):
        if self._initialized:
            return
            
        self.context = context
        self.device = device
        self._pipe = None
        self._ip_adapter_loaded = False
        self.style_buffer = StyleBuffer()
        self._initialized = True

    def warmup(self) -> None:
        if self._pipe is not None:
            return
            
        try:
            import torch
            # --- HOTFIX FOR PYTORCH 2.6.0 + TORCHAO ---
            import torch.utils._pytree
            if not hasattr(torch.utils._pytree, 'register_constant'):
                torch.utils._pytree.register_constant = lambda *args, **kwargs: None
            # ------------------------------------------
            from diffusers import StableDiffusionPipeline, LCMScheduler, AutoencoderTiny
        except ImportError:
            raise ImportError("Vui lòng cài đặt diffusers và torch")

        checkpoint = self.context.checkpoint if self.context and self.context.checkpoint else "stablediffusionapi/anything-v5"
        if checkpoint == "anything-v5":
            checkpoint = "stablediffusionapi/anything-v5"
            
        import os
        cache_dir = os.path.join(os.getcwd(), "storage", "models")
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Loading SD Pipeline with {checkpoint} into {cache_dir}")
        self._pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            safety_checker=None,
            cache_dir=cache_dir
        )
        
        logger.info("Loading TAESD VAE")
        try:
            self._pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Could not load TAESD: {e}")
            
        logger.info("Loading Hyper-SD LoRA")
        self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)
        try:
            self._pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-SD15-2steps-lora.safetensors", cache_dir=cache_dir)
            self._pipe.fuse_lora()
        except Exception as e:
            logger.warning(f"Could not load Hyper-SD LoRA: {e}")
            
        logger.info("Loading IP-Adapter FaceID")
        try:
            self._pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid_sd15.bin", cache_dir=cache_dir)
            self._ip_adapter_loaded = True
        except Exception as e:
            logger.warning(f"Could not load IP-Adapter FaceID: {e}")
            
        logger.info("Using PyTorch 2.0+ native Scaled Dot-Product Attention (SDPA)")
            
        self._pipe = self._pipe.to(self.device)
        logger.info("Pipeline warmup completed.")

    def _get_combined_embedding(self, face_embedding: Optional[np.ndarray]) -> Optional[list]:
        if face_embedding is None or not self._ip_adapter_loaded:
            return None
            
        import torch
        face_tensor = torch.tensor(face_embedding, dtype=torch.float16).to(self.device)
        if face_tensor.dim() == 1:
            face_tensor = face_tensor.unsqueeze(0)
            
        style_tensor = self.style_buffer.get_style_embedding()
        
        if style_tensor is not None:
            if face_tensor.shape == style_tensor.shape:
                combined = 0.75 * face_tensor + 0.25 * style_tensor
                return [combined]
            else:
                return [face_tensor]
        else:
            return [face_tensor]

    def generate_draft(
        self,
        prompt: str,
        negative_prompt: str,
        face_embedding: Optional[np.ndarray],
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        num_steps: int = 2,
        guidance_scale: float = 0.0
    ) -> Tuple[Image.Image, int]:
        
        self.warmup()
        import torch
        
        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()
            
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        ip_adapter_image_embeds = self._get_combined_embedding(face_embedding)
        kwargs = {}
        if ip_adapter_image_embeds is not None:
            kwargs["ip_adapter_image_embeds"] = ip_adapter_image_embeds
            
        quality_prompt = "masterpiece, best quality, highres, " + prompt
        result = self._pipe(
            prompt=quality_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            **kwargs
        ).images[0]
        
        return result, seed

    def generate_batch(
        self,
        prompt: str,
        negative_prompt: str,
        face_embedding: Optional[np.ndarray],
        batch_size: int = 4,
        seeds: Optional[List[int]] = None,
        num_steps: int = 2,
        guidance_scale: float = 0.0,
        quality_mode: bool = False
    ) -> List[Tuple[Image.Image, int]]:
        
        self.warmup()
        import torch
        
        if seeds is None or len(seeds) != batch_size:
            seeds = [torch.randint(0, 2147483647, (1,)).item() for _ in range(batch_size)]
            
        generators = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
        
        if quality_mode:
            num_steps = 8
            guidance_scale = 5.0
            try:
                self._pipe.unfuse_lora()
            except:
                pass
                
        ip_adapter_image_embeds = self._get_combined_embedding(face_embedding)
        if ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = [embed.repeat(batch_size, 1) for embed in ip_adapter_image_embeds]
            
        kwargs = {}
        if ip_adapter_image_embeds is not None:
            kwargs["ip_adapter_image_embeds"] = ip_adapter_image_embeds
            
        quality_prompt = "masterpiece, best quality, highres, " + prompt
        images = self._pipe(
            prompt=[quality_prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generators,
            **kwargs
        ).images
        
        if quality_mode:
            try:
                self._pipe.fuse_lora()
            except:
                pass
                
        return list(zip(images, seeds))

    def release(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._ip_adapter_loaded = False
            
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
            
        logger.info("StorytellingPipeline released VRAM.")
