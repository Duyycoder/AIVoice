import os
import threading
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
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
except ImportError:
    pass

from app.services.storytelling.models import StoryContext
from app.config import load_storytelling_config

# Ngưỡng steps để chuyển giữa chế độ Fast (Hyper-SD LoRA) và Quality (DPM++)
QUALITY_MODE_MIN_STEPS = 15
# Trọng số khuyến nghị cho FaceID LoRA (theo repo h94/IP-Adapter-FaceID)
FACEID_LORA_WEIGHT = 0.65
# LoRA nhân vật (train bằng scripts/train_character_lora.py, commit được vào git)
_MC_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CHARACTER_LORA_DIR = os.path.join(_MC_ROOT_DIR, "resource", "character_loras")
CHARACTER_LORA_WEIGHT = 0.8


def get_character_lora_path(char_slug: str) -> str:
    """Đường dẫn LoRA nhân vật nếu tồn tại, ngược lại trả chuỗi rỗng."""
    if not char_slug:
        return ""
    path = os.path.join(CHARACTER_LORA_DIR, f"{char_slug}.safetensors")
    return path if os.path.exists(path) else ""


class StorytellingPipeline:
    """
    Singleton quản lý Stable Diffusion pipeline cho AI Storytelling.

    Nguyên tắc đồng nhất nhân vật (identity):
    - IP-Adapter FaceID (ip-adapter-faceid_sd15.bin) nhận InsightFace embedding 512-dim.
      PHẢI load với image_encoder_folder=None vì FaceID không dùng CLIP image encoder.
    - FaceID LoRA (ip-adapter-faceid_sd15_lora.safetensors) PHẢI được kích hoạt kèm theo,
      nếu không độ giống khuôn mặt giảm mạnh.
    - Embedding truyền vào pipeline phải có shape [1, 1, 512]; khi CFG bật (guidance > 1)
      phải concat thêm zero-negative thành [2, 1, 512].
    """
    _instance = None
    _lock = threading.Lock()  # chống race condition warmup/release khi Streamlit rerun

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(StorytellingPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, context: StoryContext = None, device: str = "cuda"):
        if self._initialized:
            # Detect model change: if context has a different checkpoint, force reload
            if context and self.context:
                new_ckpt = context.checkpoint or "stablediffusionapi/anything-v5"
                old_ckpt = self.context.checkpoint or "stablediffusionapi/anything-v5"
                if new_ckpt != old_ckpt:
                    logger.info(f"Model changed from {old_ckpt} to {new_ckpt}. Releasing old pipeline...")
                    self.release()
                    self._initialized = False  # Force re-init
                else:
                    return
            else:
                return

        self.context = context
        self.device = device
        self._pipe = None
        self._ip_adapter_loaded = False
        self._ip_adapter_scale = 0.6
        self._identity_engine = "clip"       # "clip" (mọi art style) | "faceid" (mặt người thật)
        self._loaded_loras: set = set()      # adapter_name đã load vào pipeline
        self._active_mode: Optional[tuple] = None  # (mode, hyper_lora_name, char_lora) đang kích hoạt
        self._compel = None                  # encoder prompt dài >77 token (lazy init)
        self._char_lora_slug: Optional[str] = None  # LoRA nhân vật đang chọn
        self.warnings: List[str] = []        # cảnh báo cần hiển thị lên UI (không nuốt lỗi âm thầm)
        self._initialized = True

    # ------------------------------------------------------------------
    # Warmup & cấu hình chế độ
    # ------------------------------------------------------------------

    def warmup(self, num_steps: Optional[int] = None, guidance_scale: Optional[float] = None) -> None:
        """
        Load pipeline (nếu chưa) và cấu hình scheduler/LoRA theo THAM SỐ THỰC TẾ
        sẽ dùng để sinh ảnh (không dựa vào config.toml khi UI truyền tham số khác).
        """
        st_config = load_storytelling_config()
        provider = st_config.get("image_gen_provider", "Stable Diffusion (Local GPU)")
        if provider == "Local Gemini (API)":
            logger.info("Local Gemini (API) provider is active. Bypassing Stable Diffusion pipeline warmup.")
            return

        if num_steps is None:
            num_steps = st_config.get("num_inference_steps", 8)
        if guidance_scale is None:
            guidance_scale = st_config.get("guidance_scale", 1.5)

        with StorytellingPipeline._lock:
            if self._pipe is None:
                self._load_base_pipeline(st_config)
            self._configure_mode(num_steps, guidance_scale)

    def _load_base_pipeline(self, st_config: dict) -> None:
        try:
            import torch
            import torch.utils._pytree
            if not hasattr(torch.utils._pytree, 'register_constant'):
                torch.utils._pytree.register_constant = lambda *args, **kwargs: None
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError("Vui lòng cài đặt diffusers và torch")

        from app.services.storytelling.hardware_adapter import get_hardware_config
        hw_config = get_hardware_config()
        self.device = hw_config["sd_device"]
        dtype = torch.float16 if hw_config["use_fp16"] else torch.float32
        self.warnings.clear()

        checkpoint = self.context.checkpoint if self.context and self.context.checkpoint else "stablediffusionapi/anything-v5"
        if checkpoint == "anything-v5":
            checkpoint = "stablediffusionapi/anything-v5"

        # Tắt hf_transfer để tránh lỗi file lock khi download model
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        # Ưu tiên dùng HF_HOME để đồng bộ cache với toàn bộ dự án AIVoice
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            # Đường dẫn tuyệt đối theo MediaComposer root — không phụ thuộc CWD
            _mc_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            cache_dir = os.path.join(_mc_root, "storage", "models")
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_dir = cache_dir

        logger.info(f"Loading SD Pipeline with {checkpoint} into {cache_dir} (dtype={dtype})")
        self._pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            safety_checker=None,
            cache_dir=cache_dir
        )

        # ------------------------------------------------------------------
        # IDENTITY ENGINE:
        # - "clip" (mặc định): IP-Adapter Plus Face — encoder CLIP nhận THẲNG ảnh
        #   tham chiếu, hoạt động với MỌI art style (anime/manhwa/thật).
        # - "faceid": IP-Adapter FaceID — chỉ phù hợp ảnh mặt NGƯỜI THẬT
        #   (InsightFace không hiểu mặt anime).
        # ------------------------------------------------------------------
        identity_engine = st_config.get("identity_engine", "clip")
        self._identity_engine = identity_engine

        if identity_engine == "faceid":
            logger.info("Loading IP-Adapter FaceID (ip-adapter-faceid_sd15.bin)")
            try:
                self._pipe.load_ip_adapter(
                    "h94/IP-Adapter-FaceID",
                    subfolder=None,
                    weight_name="ip-adapter-faceid_sd15.bin",
                    image_encoder_folder=None,  # BẮT BUỘC: FaceID không dùng CLIP encoder
                    cache_dir=cache_dir,
                )
                self._pipe.set_ip_adapter_scale(self._ip_adapter_scale)
                self._ip_adapter_loaded = True
                logger.info("IP-Adapter FaceID loaded successfully.")
            except Exception as e:
                self._ip_adapter_loaded = False
                msg = (f"KHÔNG load được IP-Adapter FaceID: {e}. "
                       "Ảnh sẽ sinh ra KHÔNG có điều kiện khuôn mặt — nhân vật sẽ không đồng nhất!")
                logger.error(msg)
                self.warnings.append(msg)

            # FaceID LoRA đi kèm — bắt buộc để đạt độ giống khuôn mặt tốt
            if self._ip_adapter_loaded:
                try:
                    self._pipe.load_lora_weights(
                        "h94/IP-Adapter-FaceID",
                        weight_name="ip-adapter-faceid_sd15_lora.safetensors",
                        adapter_name="faceid_lora",
                        cache_dir=cache_dir,
                    )
                    self._loaded_loras.add("faceid_lora")
                    logger.info("FaceID LoRA loaded (adapter_name=faceid_lora).")
                except Exception as e:
                    msg = f"Không load được FaceID LoRA: {e}. Độ giống nhân vật sẽ giảm."
                    logger.warning(msg)
                    self.warnings.append(msg)
        else:
            logger.info("Loading IP-Adapter Plus Face (CLIP) — nhận thẳng ảnh tham chiếu, hỗ trợ anime")
            try:
                self._pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter-plus-face_sd15.safetensors",
                    cache_dir=cache_dir,
                )
                self._pipe.set_ip_adapter_scale(self._ip_adapter_scale)
                self._ip_adapter_loaded = True
                logger.info("IP-Adapter Plus Face (CLIP) loaded successfully.")
            except Exception as e:
                self._ip_adapter_loaded = False
                msg = (f"KHÔNG load được IP-Adapter Plus Face: {e}. "
                       "Ảnh sẽ sinh ra KHÔNG có điều kiện khuôn mặt — nhân vật sẽ không đồng nhất!")
                logger.error(msg)
                self.warnings.append(msg)

        logger.info("Using PyTorch 2.0+ native Scaled Dot-Product Attention (SDPA)")

        if hw_config["enable_cpu_offload"]:
            try:
                self._pipe.enable_model_cpu_offload()
                if hasattr(self._pipe, "vae") and self._pipe.vae is not None:
                    self._pipe.vae.enable_slicing()
                logger.info("Pipeline warmup completed with CPU Offload (VRAM Save).")
            except Exception as e:
                logger.warning(f"Không thể bật CPU Offload: {e}. Fallback load trực tiếp model lên thiết bị: {self.device}")
                self._pipe = self._pipe.to(self.device)
        else:
            self._pipe = self._pipe.to(self.device)
            logger.info(f"Pipeline warmup completed directly on device {self.device} (Maximum Speed).")

    def _select_hyper_lora(self, num_steps: int, guidance_scale: float) -> str:
        if num_steps <= 2:
            return "Hyper-SD15-2steps-lora.safetensors"
        if num_steps <= 4:
            return "Hyper-SD15-4steps-lora.safetensors"
        if guidance_scale > 1.2:
            return "Hyper-SD15-8steps-CFG-lora.safetensors"
        return "Hyper-SD15-8steps-lora.safetensors"

    def set_character_lora(self, char_slug: Optional[str]) -> None:
        """
        Chọn LoRA nhân vật cho các lần sinh tiếp theo (None = không dùng).
        LoRA được load lazy trong _configure_mode; đổi nhân vật chỉ swap adapter,
        không reload model.
        """
        slug = char_slug if get_character_lora_path(char_slug or "") else None
        if slug != self._char_lora_slug:
            self._char_lora_slug = slug
            self._active_mode = None  # buộc _configure_mode chạy lại để swap adapter

    def _configure_mode(self, num_steps: int, guidance_scale: float) -> None:
        """
        Đồng bộ scheduler + LoRA với tham số sinh ảnh THỰC TẾ.
        Dùng set_adapters (bật/tắt) thay cho fuse/unfuse để chuyển chế độ an toàn,
        không làm hỏng trọng số UNet khi chuyển qua lại Fast <-> Quality.
        """
        if self._pipe is None:
            return

        if num_steps >= QUALITY_MODE_MIN_STEPS:
            mode_key = ("quality", None, self._char_lora_slug)
        else:
            hyper_name = self._select_hyper_lora(num_steps, guidance_scale)
            mode_key = ("fast", hyper_name, self._char_lora_slug)

        if mode_key == self._active_mode:
            return

        scheduler_config = dict(self._pipe.scheduler.config)
        scheduler_config.pop("algorithm_type", None)
        scheduler_config.pop("final_sigmas_type", None)

        active_adapters = []
        adapter_weights = []
        if "faceid_lora" in self._loaded_loras:
            active_adapters.append("faceid_lora")
            adapter_weights.append(FACEID_LORA_WEIGHT)

        if mode_key[0] == "quality":
            logger.info("Quality Mode: DPMSolverMultistepScheduler (Karras), no Hyper-SD LoRA")
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                scheduler_config, use_karras_sigmas=True
            )
        else:
            hyper_name = mode_key[1]
            adapter_name = "hyper_" + hyper_name.replace("-", "_").replace(".safetensors", "").lower()
            logger.info(f"Fast Mode: EulerDiscreteScheduler + {hyper_name}")
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(scheduler_config)

            if adapter_name not in self._loaded_loras:
                try:
                    self._pipe.load_lora_weights(
                        "ByteDance/Hyper-SD",
                        weight_name=hyper_name,
                        adapter_name=adapter_name,
                        cache_dir=getattr(self, "_cache_dir", None),
                    )
                    self._loaded_loras.add(adapter_name)
                except Exception as e:
                    msg = f"Không load được Hyper-SD LoRA ({hyper_name}): {e}. Chất lượng ảnh fast-mode sẽ kém."
                    logger.warning(msg)
                    self.warnings.append(msg)
            if adapter_name in self._loaded_loras:
                active_adapters.append(adapter_name)
                adapter_weights.append(1.0)

        # LoRA nhân vật (nếu được chọn qua set_character_lora)
        if self._char_lora_slug:
            char_adapter = f"char_{self._char_lora_slug}"
            if char_adapter not in self._loaded_loras:
                lora_path = get_character_lora_path(self._char_lora_slug)
                try:
                    self._pipe.load_lora_weights(lora_path, adapter_name=char_adapter)
                    self._loaded_loras.add(char_adapter)
                    logger.info(f"Đã load LoRA nhân vật: {lora_path}")
                except Exception as e:
                    msg = f"Không load được LoRA nhân vật '{self._char_lora_slug}': {e}"
                    logger.warning(msg)
                    self.warnings.append(msg)
            if char_adapter in self._loaded_loras:
                active_adapters.append(char_adapter)
                adapter_weights.append(CHARACTER_LORA_WEIGHT)

        try:
            if self._loaded_loras:
                self._pipe.set_adapters(active_adapters, adapter_weights)
        except Exception as e:
            logger.warning(f"set_adapters failed: {e}")

        self._active_mode = mode_key
        logger.info(f"Pipeline mode configured: {mode_key[0]} (steps={num_steps}, cfg={guidance_scale}), "
                    f"active LoRAs: {active_adapters}")

    # ------------------------------------------------------------------
    # IP-Adapter helpers
    # ------------------------------------------------------------------

    def update_ip_adapter_scale(self, scale: float) -> None:
        """Cho phép user tinh chỉnh IP-Adapter scale (mức giống nhân vật) runtime."""
        self._ip_adapter_scale = max(0.0, min(1.0, scale))
        if self._pipe and self._ip_adapter_loaded:
            try:
                self._pipe.set_ip_adapter_scale(self._ip_adapter_scale)
                logger.info(f"IP-Adapter scale updated to {self._ip_adapter_scale}")
            except Exception as e:
                logger.warning(f"Could not set IP-Adapter scale: {e}")

    def _prepare_ip_embeds(
        self,
        face_embedding: Optional[np.ndarray],
        guidance_scale: float,
    ) -> Optional[list]:
        """
        Chuẩn hoá InsightFace embedding về đúng format mà diffusers yêu cầu:
        - [1, 1, 512] khi CFG tắt (guidance <= 1)
        - [2, 1, 512] = concat(zero-negative, positive) khi CFG bật (guidance > 1)
        dtype/device lấy từ chính pipeline (không hardcode fp16 → không vỡ khi chạy CPU).
        """
        if face_embedding is None or not self._ip_adapter_loaded:
            return None

        import torch
        dtype = self._pipe.unet.dtype
        emb = torch.as_tensor(np.asarray(face_embedding), dtype=dtype, device=self.device)
        emb = emb.reshape(1, 1, -1)  # [1, 1, 512]

        if guidance_scale > 1.0:
            # Diffusers sẽ chunk(2) để tách negative/positive khi CFG bật
            emb = torch.cat([torch.zeros_like(emb), emb], dim=0)  # [2, 1, 512]

        return [emb]

    def _build_identity_kwargs(
        self,
        face_embedding: Optional[np.ndarray],
        face_image: Optional[Image.Image],
        guidance_scale: float,
    ) -> dict:
        """
        Chọn input identity đúng theo engine đang load:
        - clip  : truyền THẲNG ảnh tham chiếu (ip_adapter_image) — hỗ trợ anime.
        - faceid: truyền InsightFace embedding (ip_adapter_image_embeds).
        Khi cảnh KHÔNG có nhân vật: vẫn phải truyền input trung tính với scale=0
        (IP-Adapter đã gắn vào UNet nên không thể bỏ trống input).
        """
        if not self._ip_adapter_loaded:
            return {}

        engine = getattr(self, "_identity_engine", "clip")

        if engine == "clip":
            if face_image is not None:
                self._pipe.set_ip_adapter_scale(self._ip_adapter_scale)
                return {"ip_adapter_image": face_image}
            # Không có ảnh tham chiếu → vô hiệu hoá IP-Adapter cho lần sinh này
            self._pipe.set_ip_adapter_scale(0.0)
            return {"ip_adapter_image": Image.new("RGB", (224, 224), (128, 128, 128))}

        # engine == "faceid"
        if face_embedding is not None:
            self._pipe.set_ip_adapter_scale(self._ip_adapter_scale)
            return {"ip_adapter_image_embeds": self._prepare_ip_embeds(face_embedding, guidance_scale)}
        self._pipe.set_ip_adapter_scale(0.0)
        zero_emb = np.zeros(512, dtype=np.float32)
        return {"ip_adapter_image_embeds": self._prepare_ip_embeds(zero_emb, guidance_scale)}

    @staticmethod
    def _a1111_to_compel(prompt: str) -> str:
        """
        Chuyển cú pháp trọng số A1111 `(tag:1.3)` → cú pháp compel `(tag)1.3`.
        Nhóm không có trọng số `(tag)` giữ nguyên (compel hiểu là nhấn nhẹ 1.1).
        """
        import re
        return re.sub(r"\(([^()]+):([0-9.]+)\)", r"(\1)\2", prompt or "")

    @staticmethod
    def _strip_weights(prompt: str) -> str:
        """Bỏ toàn bộ cú pháp trọng số — dùng cho fallback khi compel không khả dụng
        (để `(tag:1.3)` không thành token rác trong CLIP)."""
        import re
        s = re.sub(r"\(([^()]+):[0-9.]+\)", r"\1", prompt or "")   # (tag:1.3) -> tag
        s = re.sub(r"\(([^()]+)\)([0-9.]+)?", r"\1", s)             # (tag)1.3 / (tag) -> tag
        return s

    def _build_prompt_kwargs(self, prompt: str, negative_prompt: str) -> dict:
        """
        T3: LUÔN encode prompt qua compel (không chỉ khi >77 token) vì 2 lý do:
        1. Trọng số `(tag:1.3)` chỉ có tác dụng khi đi qua compel — truyền chuỗi
           thường vào diffusers thì cú pháp này thành TOKEN RÁC trong CLIP.
        2. Prompt >77 token được nối embedding, không bị cắt.
        Fallback (compel thiếu/lỗi): strip sạch cú pháp trọng số rồi truyền thường.
        """
        try:
            from compel import Compel
        except ImportError:
            msg = ("Thư viện 'compel' chưa được cài — trọng số prompt (tag:1.3) bị bỏ qua "
                   "và prompt >77 token sẽ bị CLIP cắt. Chạy: pip install compel")
            logger.warning(msg)
            if msg not in self.warnings:
                self.warnings.append(msg)
            return {"prompt": self._strip_weights(prompt),
                    "negative_prompt": self._strip_weights(negative_prompt)}

        try:
            prompt = self._a1111_to_compel(prompt)
            negative_prompt = self._a1111_to_compel(negative_prompt or "")
            if not hasattr(self, "_compel") or self._compel is None:
                # Với enable_model_cpu_offload(), text_encoder.device báo "cpu" lúc idle
                # nhưng hook sẽ đẩy nó lên GPU khi forward → phải ép device về
                # execution device để token ids nằm cùng device với weights.
                exec_device = getattr(self._pipe, "_execution_device", None) or self.device
                self._compel = Compel(
                    tokenizer=self._pipe.tokenizer,
                    text_encoder=self._pipe.text_encoder,
                    truncate_long_prompts=False,
                    device=str(exec_device),
                )
            pos_embeds = self._compel(prompt)
            neg_embeds = self._compel(negative_prompt or "")
            [pos_embeds, neg_embeds] = self._compel.pad_conditioning_tensors_to_same_length(
                [pos_embeds, neg_embeds]
            )
            return {"prompt_embeds": pos_embeds, "negative_prompt_embeds": neg_embeds}
        except Exception as e:
            logger.warning(f"Compel encode lỗi: {e}. Fallback về prompt thường (trọng số bị bỏ, >77 token bị cắt).")
            return {"prompt": self._strip_weights(prompt),
                    "negative_prompt": self._strip_weights(negative_prompt)}

    # ------------------------------------------------------------------
    # Gemini API provider
    # ------------------------------------------------------------------

    def _generate_via_gemini_api(self, prompt: str) -> Image.Image:
        from openai import OpenAI
        from app.config import config
        import requests
        from io import BytesIO

        api_key = config.app.get("openai_api_key", "")
        base_url = config.app.get("openai_base_url", "http://localhost:7860/v1")
        if not api_key:
            raise ValueError("Chưa cấu hình openai_api_key trong Global Settings (cần cho Local Gemini API).")

        logger.info(f"Generating image via Gemini API: {prompt} (base_url: {base_url})")
        client = OpenAI(api_key=api_key, base_url=base_url)

        try:
            response = client.images.generate(
                model="gemini-2.0-flash",
                prompt=prompt,
                n=1
            )
            url = response.data[0].url
        except Exception as e:
            logger.error(f"Error calling Gemini image generation API: {e}")
            raise e

        if not url:
            raise ValueError("No image URL returned from Gemini API")

        logger.info(f"Gemini API returned image URL: {url}")

        img_response = requests.get(url, timeout=60)
        img_response.raise_for_status()

        return Image.open(BytesIO(img_response.content)).convert("RGB")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_draft(
        self,
        prompt: str,
        negative_prompt: str,
        face_embedding: Optional[np.ndarray],
        seed: int = -1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        face_image: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, int]:

        st_config = load_storytelling_config()
        provider = st_config.get("image_gen_provider", "Stable Diffusion (Local GPU)")
        if provider == "Local Gemini (API)":
            img = self._generate_via_gemini_api(prompt)
            return img, seed if seed != -1 else 0

        import torch

        if width is None:
            width = st_config.get("image_width", 896)
        if height is None:
            height = st_config.get("image_height", 512)
        if num_steps is None:
            num_steps = st_config.get("num_inference_steps", 2)
        if guidance_scale is None:
            guidance_scale = st_config.get("guidance_scale", 0.0)

        # Warmup với THAM SỐ THỰC TẾ để scheduler/LoRA khớp chế độ sinh ảnh
        self.warmup(num_steps=num_steps, guidance_scale=guidance_scale)

        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()

        generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = self._build_identity_kwargs(face_embedding, face_image, guidance_scale)

        quality_prompt = "masterpiece, best quality, highres, " + prompt
        prompt_kwargs = self._build_prompt_kwargs(quality_prompt, negative_prompt)
        result = self._pipe(
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            **prompt_kwargs,
            **kwargs
        ).images[0]

        self.log_vram("sau generate")
        return result, seed

    def generate_batch(
        self,
        prompt: str,
        negative_prompt: str,
        face_embedding: Optional[np.ndarray],
        batch_size: int = 4,
        seeds: Optional[List[int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        quality_mode: bool = False,
        face_image: Optional[Image.Image] = None
    ) -> List[Tuple[Image.Image, int]]:

        st_config = load_storytelling_config()
        provider = st_config.get("image_gen_provider", "Stable Diffusion (Local GPU)")
        if provider == "Local Gemini (API)":
            import random
            if seeds is None or len(seeds) != batch_size:
                seeds = [random.randint(0, 2147483647) for _ in range(batch_size)]

            logger.info(f"Generating {batch_size} images via Gemini API (batch generation)")
            from openai import OpenAI
            from app.config import config
            import requests
            from io import BytesIO

            api_key = config.app.get("openai_api_key", "")
            base_url = config.app.get("openai_base_url", "http://localhost:7860/v1")
            if not api_key:
                raise ValueError("Chưa cấu hình openai_api_key trong Global Settings (cần cho Local Gemini API).")

            client = OpenAI(api_key=api_key, base_url=base_url)

            try:
                response = client.images.generate(
                    model="gemini-2.0-flash",
                    prompt=prompt,
                    n=batch_size
                )
                urls = [item.url for item in response.data if item.url]
            except Exception as e:
                logger.error(f"Failed to generate batch from Gemini API: {e}")
                urls = []

            images = []
            for i, url in enumerate(urls[:batch_size]):
                try:
                    logger.info(f"Downloading batch image {i+1}/{len(urls)}: {url}")
                    img_response = requests.get(url, timeout=60)
                    img_response.raise_for_status()
                    img = Image.open(BytesIO(img_response.content)).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.error(f"Failed to download image from {url}: {e}")

            while len(images) < batch_size:
                logger.warning(f"Batch only returned {len(images)}/{batch_size} images. Generating extra image sequentially...")
                try:
                    img = self._generate_via_gemini_api(prompt)
                    images.append(img)
                except Exception as e:
                    logger.error(f"Failed to generate sequential fallback image: {e}")
                    break

            return list(zip(images, seeds[:len(images)]))

        import torch

        if width is None:
            width = st_config.get("image_width", 896)
        if height is None:
            height = st_config.get("image_height", 512)
        if num_steps is None:
            num_steps = st_config.get("num_inference_steps", 2)
        if guidance_scale is None:
            guidance_scale = st_config.get("guidance_scale", 0.0)

        if quality_mode:
            num_steps = max(num_steps, 25)
            guidance_scale = 7.0

        # Warmup + cấu hình chế độ theo tham số thực tế (thay cho hack unfuse/fuse LoRA cũ)
        self.warmup(num_steps=num_steps, guidance_scale=guidance_scale)

        if seeds is None or len(seeds) != batch_size:
            seeds = [torch.randint(0, 2147483647, (1,)).item() for _ in range(batch_size)]

        generators = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]

        kwargs = self._build_identity_kwargs(face_embedding, face_image, guidance_scale)

        quality_prompt = "masterpiece, best quality, highres, " + prompt
        prompt_kwargs = self._build_prompt_kwargs(quality_prompt, negative_prompt)
        images = []

        logger.info(f"Generating {batch_size} images sequentially to save VRAM...")
        for i in range(batch_size):
            img = self._pipe(
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generators[i],
                **prompt_kwargs,
                **kwargs
            ).images[0]
            images.append(img)

        return list(zip(images, seeds))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def log_vram(tag: str = "") -> None:
        """Log tình trạng VRAM — phát hiện tràn (spill sang RAM làm chậm 2-3x)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            pct = reserved / total * 100
            msg = f"[VRAM{' ' + tag if tag else ''}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB / {total:.2f}GB ({pct:.0f}%)"
            if pct > 92:
                logger.warning(msg + " ⚠️ SÁT TRẦN — driver có thể đang spill sang RAM (chậm 2-3x). "
                                     "Cân nhắc hardware_profile=auto hoặc tắt bớt tính năng.")
            else:
                logger.info(msg)
        except Exception:
            pass

    def free_vram_cache(self) -> None:
        """Dọn rác VRAM tạm thời mà KHÔNG hủy model (dùng giữa các batch)."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def release(self) -> None:
        with StorytellingPipeline._lock:
            self._release_unlocked()

    def _release_unlocked(self) -> None:
        if self._pipe is not None:
            # Unload lora weights if possible to break references
            try:
                self._pipe.unload_lora_weights()
            except Exception:
                pass

            # Xóa _pipe thay vì can thiệp vào các module con vì CPU offload hook có thể bị break
            del self._pipe
            self._pipe = None
            self._ip_adapter_loaded = False
            self._loaded_loras.clear()
            self._active_mode = None
            self._compel = None
            self._img2img = None  # cache img2img của Face Detailer (chia sẻ component)

        import gc
        # Gọi gc.collect() nhiều lần để đảm bảo Python dọn dẹp sạch cyclic references
        gc.collect()
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                # BẮT BUỘC: Đồng bộ hóa toàn bộ luồng GPU trước khi xóa bộ nhớ đệm
                # Tránh lỗi CUDA illegal memory access khi kernel đang chạy ngầm
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

        logger.info("StorytellingPipeline VRAM đã được giải phóng hoàn toàn.")
# EOF
