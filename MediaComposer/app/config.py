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
                if "proxy" in data:
                    self.proxy = data["proxy"]

    def save_config(self):
        data = {
            "app": self.app,
            "whisper": self.whisper
        }
        if self.proxy is not None:
            data["proxy"] = self.proxy
        with open(self.config_file, "w", encoding="utf-8") as f:
            toml.dump(data, f)

config = Config()
