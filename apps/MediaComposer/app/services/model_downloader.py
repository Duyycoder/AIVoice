import os
import sys
import requests
from loguru import logger
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# MediaComposer root directory
MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(MC_ROOT, "models")

# QUAN TRỌNG: phải TRÙNG với cache_dir của image_generator lúc RUNTIME
# (run.bat KHÔNG set HF_HOME → runtime dùng storage/models). Trước đây setup tải
# vào models/diffusers_cache hoặc HF_HOME của setup.bat → runtime không thấy,
# tải lại toàn bộ lần nữa. Hardcode storage/models để tải 1 lần dùng mọi nơi.
CACHE_DIR = os.path.join(MC_ROOT, "storage", "models")

# Danh sách model RUNTIME THẬT SỰ cần (đồng bộ với image_generator/face_detailer
# sau đợt nâng cấp 07/2026: engine mặc định = IP-Adapter Plus Face CLIP)
REQUIRED_MODELS = {
    "realesrgan_anime": {
        "type": "url",
        "path": os.path.join(MODELS_DIR, "realesrgan", "RealESRGAN_x4plus_anime_6B.pth"),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "desc": "RealESRGAN 4x Anime Upscaler"
    },
    "realesrgan_video": {
        "type": "url",
        "path": os.path.join(MODELS_DIR, "realesrgan", "realesr-animevideov3.pth"),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "desc": "RealESRGAN AnimeVideo v3 (nhẹ, nhanh)"
    },
    "anime_face_detector": {
        "type": "url",
        "path": os.path.join(MC_ROOT, "resource", "models", "lbpcascade_animeface.xml"),
        "url": "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml",
        "desc": "Anime Face Detector (Face Detailer + dataset gate)",
        "min_size": 50000,
    },
    "ip_adapter_plus_face": {
        "type": "hf",
        "repo_id": "h94/IP-Adapter",
        "filename": "models/ip-adapter-plus-face_sd15.safetensors",
        "desc": "IP-Adapter Plus Face (engine mặc định, hỗ trợ anime)"
    },
    "ip_adapter_image_encoder": {
        "type": "hf",
        "repo_id": "h94/IP-Adapter",
        "filename": "models/image_encoder/model.safetensors",
        "desc": "CLIP ViT-H Image Encoder cho IP-Adapter (~2.5GB, tải 1 lần)"
    },
    "ip_adapter_image_encoder_cfg": {
        "type": "hf",
        "repo_id": "h94/IP-Adapter",
        "filename": "models/image_encoder/config.json",
        "desc": "Config CLIP Image Encoder"
    },
    "hyper_sd_8step_cfg": {
        "type": "hf",
        "repo_id": "ByteDance/Hyper-SD",
        "filename": "Hyper-SD15-8steps-CFG-lora.safetensors",
        "desc": "Hyper-SD 8-Step CFG LoRA (chế độ nhanh mặc định)"
    },
}

def _download_file(url: str, dest_path: str, desc: str = "", progress_callback = None) -> bool:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        logger.info(f"Downloading {desc} from {url} ...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            file=sys.stdout
        ) as bar:
            downloaded = 0
            for data in response.iter_content(chunk_size=1024 * 64):
                size = file.write(data)
                bar.update(size)
                downloaded += size
                if progress_callback and total_size > 0:
                    pct = int((downloaded / total_size) * 100)
                    progress_callback(f"Đang tải {desc}...", pct)
                    
        logger.info(f"Successfully downloaded {desc} to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {desc}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        return False

def ensure_models_ready(config=None, download_if_missing: bool = True, progress_callback = None) -> dict:
    """
    Kiểm tra từng model, tải nếu thiếu. Trả về {model_name: status}.
    Status có thể là 'ready', 'missing', hoặc 'error'.
    """
    results = {}
    for key, info in REQUIRED_MODELS.items():
        desc = info.get("desc", key)
        if info["type"] == "url":
            path = info["path"]
            min_size = info.get("min_size", 1000000)
            if os.path.exists(path) and os.path.getsize(path) > min_size:
                results[key] = "ready"
            else:
                if download_if_missing:
                    if progress_callback:
                        progress_callback(f"Đang tải model: {desc}...", 0)
                    success = _download_file(info["url"], path, desc=desc, progress_callback=progress_callback)
                    results[key] = "ready" if success else "error"
                else:
                    results[key] = "missing"
        elif info["type"] == "hf":
            if not hf_hub_download:
                logger.warning("huggingface_hub module missing, cannot verify HF models.")
                results[key] = "error"
                continue
            try:
                # Check/download via hf_hub_download
                os.makedirs(CACHE_DIR, exist_ok=True)
                local_only = not download_if_missing
                if progress_callback and download_if_missing:
                    progress_callback(f"Đang kiểm tra/tải model HF: {desc}...", 0)
                try:
                    res_path = hf_hub_download(
                        repo_id=info["repo_id"],
                        filename=info["filename"],
                        cache_dir=CACHE_DIR,
                        local_files_only=local_only
                    )
                    if res_path and os.path.exists(res_path):
                        results[key] = "ready"
                    else:
                        results[key] = "missing"
                except Exception as e:
                    if not download_if_missing:
                        results[key] = "missing"
                    else:
                        logger.error(f"Failed HF download for {desc}: {e}")
                        results[key] = "error"
            except Exception as e:
                results[key] = "error"
                
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MediaComposer Model Downloader")
    parser.add_argument("--check-only", action="store_true", help="Chỉ kiểm tra model, không tải")
    parser.add_argument("--download", action="store_true", help="Tải model nếu thiếu")
    args = parser.parse_args()
    
    download_if_missing = not args.check_only
    logger.info(f"Running model verification (download_if_missing={download_if_missing})...")
    
    status = ensure_models_ready(download_if_missing=download_if_missing)
    logger.info(f"Model status: {status}")
    
    missing_or_error = [k for k, v in status.items() if v != "ready"]
    if missing_or_error:
        logger.warning(f"Các model chưa sẵn sàng: {missing_or_error}")
        sys.exit(1)
    else:
        logger.info("Tất cả model cho Storytelling đã sẵn sàng!")
        sys.exit(0)

if __name__ == "__main__":
    main()
