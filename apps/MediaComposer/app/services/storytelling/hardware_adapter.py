import torch
from app.config import load_storytelling_config
from loguru import logger

def get_hardware_profile() -> str:
    """
    Lấy profile phần cứng từ file config.toml.
    Mặc định là "auto".
    """
    try:
        config = load_storytelling_config()
        return config.get("hardware_profile", "auto")
    except Exception:
        return "auto"

def get_hardware_config() -> dict:
    """
    Trả về cấu hình phần cứng tối ưu cho các mô hình AI dựa trên profile hoặc tự động phát hiện.
    
    Returns:
        dict: Cấu hình thiết bị chạy và các cờ tối ưu hóa cho từng model.
    """
    profile = get_hardware_profile()
    cuda_available = torch.cuda.is_available()
    
    # Thiết lập mặc định chạy CPU hoàn toàn
    config = {
        "device": "cpu",
        "sd_device": "cpu",
        "face_device": "cpu",
        "esrgan_device": "cpu",
        "whisper_device": "cpu",
        "enable_cpu_offload": False,
        "use_fp16": False,
        "profile_name": "Chỉ chạy CPU (CPU Only)"
    }
    
    if not cuda_available:
        logger.info("Không phát hiện GPU CUDA khả dụng. Sử dụng cấu hình chạy CPU.")
        return config

    # Phân giải chế độ "auto" dựa trên dung lượng VRAM thực tế
    resolved_profile = profile
    if profile == "auto":
        try:
            device_idx = torch.cuda.current_device()
            vram_bytes = torch.cuda.get_device_properties(device_idx).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            logger.info(f"Dò tìm phần cứng tự động: VRAM GPU khả dụng là {vram_gb:.2f} GB")
            if vram_gb >= 7.0:  # RTX 5060 8GB (hoặc các dòng GPU >= 8GB VRAM)
                resolved_profile = "cuda_high"
            else:
                resolved_profile = "cuda_low"
        except Exception as e:
            logger.warning(f"Lỗi khi dò tìm thông tin VRAM: {e}. Fallback về cuda_low.")
            resolved_profile = "cuda_low"

    if resolved_profile == "cuda_high":
        config.update({
            "device": "cuda",
            "sd_device": "cuda",
            "face_device": "cuda",      # Chạy InsightFace trên CUDA để tối ưu tốc độ trích xuất khuôn mặt
            "esrgan_device": "cuda",
            "whisper_device": "cuda",
            "enable_cpu_offload": False, # Tắt CPU Offload để giữ model trên VRAM (chạy cực nhanh)
            "use_fp16": True,
            "profile_name": "Tối đa tốc độ (GPU >= 8GB VRAM)"
        })
    elif resolved_profile == "cuda_low":
        config.update({
            "device": "cuda",
            "sd_device": "cuda",
            "face_device": "cpu",       # InsightFace chạy CPU để tránh tranh chấp VRAM khi vẽ ảnh
            "esrgan_device": "cuda",
            "whisper_device": "cpu",    # Whisper chạy trên CPU để chừa VRAM cho SD / RealESRGAN
            "enable_cpu_offload": True,  # Bật CPU Offload cho Stable Diffusion để tránh tràn VRAM OOM
            "use_fp16": True,
            "profile_name": "Tiết kiệm VRAM (GPU <= 6GB VRAM)"
        })
    else:  # cpu
        pass  # Giữ cấu hình chạy CPU mặc định
        
    logger.info(f"Cấu hình tăng tốc phần cứng đang kích hoạt: {config['profile_name']} (Thiết bị: {config['device']}, CPU Offload: {config['enable_cpu_offload']})")
    return config
