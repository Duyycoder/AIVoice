import os
import toml
import torch

class Config:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")
        self.app = {
            "pexels_api_keys": [],
            "pixabay_api_keys": [],
            "coverr_api_keys": [],
            "openai_api_key": "",
            "openai_base_url": "https://api.openai.com/v1",
            "openai_model": "gpt-4o-mini",
            "llm_provider": "OpenAI",
        }
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        default_compute = "float16" if default_device == "cuda" else "int8"
        self.whisper = {
            "model_size": "base",
            "device": default_device,
            "compute_type": default_compute
        }
        self.storytelling = {
            "aspect_ratio": "16:9",
            "image_width": 768,
            "image_height": 432,
            "enable_upscaling": True,
            "upscaler_model": "RealESRGAN_x4plus_anime_6B",
            "output_width": 1920,
            "output_height": 1080,
            "video_fps": 24,
            "subtitle_font": "Arial",
            "subtitle_font_size": 28,
            "subtitle_color": "white",
            "subtitle_border": 2,
            "subtitle_position": "bottom",
            "num_inference_steps": 8,
            "guidance_scale": 1.5,
            "hardware_profile": "auto",
            "image_gen_provider": "Stable Diffusion (Local GPU)",
            # "clip": IP-Adapter Plus Face — truyền thẳng ảnh tham chiếu, hỗ trợ anime/manhwa (khuyên dùng)
            # "faceid": IP-Adapter FaceID — chỉ hiệu quả với ảnh mặt người thật
            "identity_engine": "clip",
            # Face Detailer: tự động vẽ lại mặt sau khi sinh (chạy cả trong batch)
            "enable_face_detailer": True,
            "face_detailer_strength": 0.45,
            # Số mặt tối đa vẽ lại mỗi ảnh (1 = chỉ mặt nhân vật chính — nhanh gấp đôi khi có poster/nhân vật phụ)
            "face_detailer_max_faces": 1,
            # Steps cho bước vẽ mặt (14 đủ đẹp với DPM++ ở strength 0.45; 22 chỉ nhỉnh hơn chút mà chậm +60%)
            "face_detailer_steps": 14,
            # Bỏ qua vẽ lại nếu mặt draft ĐÃ đủ giống nhân vật (CLIP sim >= ngưỡng) và đủ lớn
            "face_detailer_skip_sim": 0.62,
            # Mức độ giống nhân vật (IP-Adapter) — nguồn sự thật chung cho Studio + Batch
            "ip_adapter_scale": 0.6,
        }
        self.proxy = None
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = toml.load(f)
                if "app" in data:
                    self.app.update(data["app"])
                if "whisper" in data:
                    self.whisper.update(data["whisper"])
                if "storytelling" in data:
                    self.storytelling.update(data["storytelling"])
                if "proxy" in data:
                    self.proxy = data["proxy"]

    def save_config(self):
        data = {
            "app": self.app,
            "whisper": self.whisper,
            "storytelling": self.storytelling
        }
        if self.proxy is not None:
            data["proxy"] = self.proxy
        with open(self.config_file, "w", encoding="utf-8") as f:
            toml.dump(data, f)

config = Config()

def load_storytelling_config() -> dict:
    config.load_config()
    return config.storytelling

