import os
import toml

try:
    import torch
except ImportError:  # Cho phép import config trong CI/test GPU-free.
    torch = None

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
            "llm_api_key": "",
            "llm_base_url": "",
            "llm_model": "",
        }
        default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
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
            # 5.0 chứ KHÔNG phải 1.5. Với 8 bước + Hyper-SD CFG-lora, guidance 1.5
            # gần như bỏ qua prompt: model tự do liên tưởng và trả về mảng texture
            # trừu tượng thay vì chủ thể. Ảnh nền vẫn "trông được" nên lỗi này ẩn
            # rất lâu — chỉ lộ ra ở ảnh nhân vật. config.toml.example luôn ghi 5.0;
            # mặc định 1.5 ở đây là giá trị cũ sót lại, không khớp tài liệu.
            "guidance_scale": 5.0,
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
            # ----------------------------------------------------------------
            # STYLE LOCK — khóa phong cách ở mức TRỌNG SỐ, không chỉ mức prompt.
            # Tên file (không đuôi) trong resource/style_loras/, train bằng
            # scripts/train_style_lora.py. Rỗng = tắt (chỉ khóa bằng prompt).
            # ----------------------------------------------------------------
            # Mặc định BẬT LoRA thủy mặc đã train 24/07 (46 ảnh Met CC0).
            # Trọng số 0.45 theo sweep: 0.40-0.55 dùng được, >=0.70 tan chi tiết
            # bàn tay/bàn chân, <=0.25 gần như không khác base model.
            # Đặt rỗng để tắt style lock ở mức trọng số.
            "style_lora": "thuy_mac",
            "style_lora_weight": 0.45,
            # Trọng số gắn cho cụm hành động ở đầu prompt (1.25-1.45 hợp lý;
            # cao hơn dễ méo giải phẫu, thấp hơn thì nhân vật lại đứng yên).
            "studio_action_weight": 1.35,
            # ----------------------------------------------------------------
            # STUDIO COMPOSITING (nhánh feat/studio-compositing) — render theo lớp.
            # Mặc định "classic" ⇒ giữ nguyên hành vi cũ; đặt "studio" để bật.
            # ----------------------------------------------------------------
            "render_mode": "classic",              # "classic" | "studio"
            "studio_bg_cache": True,               # tái dùng nền theo location
            "studio_layout_source": "llm",         # LLM nếu hợp lệ, tự fallback heuristic
            # Engine tách nền lớp nhân vật: "rembg" (isnet-anime, segment nhân vật —
            # bền nhất, khuyên dùng) | "chroma" (chroma-key cũ, phụ thuộc nền phẳng).
            "studio_matte_engine": "rembg",
            "studio_matte_model": "isnet-anime",   # model rembg (isnet-anime hợp anime)
            # Xám trung tính: rembg không cần chroma và xám tránh green spill lên áo.
            # (Đổi về "#00B140" nếu chuyển studio_matte_engine="chroma".)
            "studio_matte_bg_color": "#9CA3AF",    # màu nền phẳng sinh nhân vật
            "studio_matte_threshold": 0.18,        # ngưỡng khoảng cách màu (0-1) tách nền
            "studio_matte_feather_px": 3,          # làm mềm mép alpha
            "studio_matte_despill": True,          # khử ám màu nền ở viền
            "studio_matte_adaptive_fallback": True, # GrabCut nhanh nếu SD bỏ qua màu chroma
            "studio_char_use_detailer": True,      # detailer chỉ chạy ở lớp nhân vật
            "studio_char_use_ip_adapter": True,    # dùng ảnh ref (IP-Adapter) cho nhân vật
            # Auto-fallback về classic cho cảnh khó
            "studio_fallback_max_chars": 3,        # > ngưỡng → render classic cả cảnh
            "studio_fallback_interaction_tags": [
                "hug", "embrace", "fight", "holding hands", "carry", "kiss"
            ],
            "studio_shadow_opacity": 0.0,          # 0 = tắt; >0 vẽ bóng chân nhẹ
            # Tự train LoRA cho nhân vật chính (nam/nữ chính) 1 LẦN khi render truyện
            # mới: bootstrap dataset từ IP-Adapter rồi train. Idempotent (đã có LoRA
            # thì bỏ qua) nên các chương sau của truyện dùng lại, không train lại.
            "studio_auto_train_leads": True,
            "studio_auto_train_max_leads": 2,      # số nhân vật chính train tự động
            "studio_auto_train_steps": 700,        # số bước train mỗi nhân vật
            # Nhân vật ĐÃ có LoRA thì bỏ qua face detailer (LoRA đã giữ nhận dạng) —
            # tiết kiệm ~40% thời gian sinh mỗi nhân vật chính, chất lượng vẫn ổn.
            "studio_lora_skip_detailer": True,
            # Chất lượng render Studio (nền + nhân vật). >0 sẽ ghi đè num_inference_steps
            # cho riêng Studio: 8 = nhanh/phẳng (anything-v5), 18-22 = nét/chi tiết
            # (khuyên dùng checkpoint dreamshaper-8 để tiệm cận nanobanana). 0 = theo
            # num_inference_steps chung.
            "studio_render_steps": 0,
            "studio_render_guidance": 0.0,
            # Unify pass: img2img strength thấp trên frame ĐÃ ghép để nhân vật ăn
            # sáng với nền (hết vẻ dán sticker). Xem studio/unify_pass.py.
            "studio_unify_pass": True,
            "studio_unify_strength": 0.28,   # >0.45 sẽ vẽ lại cả bố cục (bị chặn)
            "studio_unify_steps": 16,        # số bước thật = steps * strength
            "studio_unify_guidance": 5.0,
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

