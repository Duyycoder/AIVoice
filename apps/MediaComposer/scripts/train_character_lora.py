# -*- coding: utf-8 -*-
"""
Train LoRA nhân vật cho AI Storytelling (Workflow 5).

Cách dùng (chạy từ thư mục apps/MediaComposer, dùng venv của dự án):
    ..\\..\\..venv\\Scripts\\python.exe scripts\\train_character_lora.py ^
        --character d_ch_phong ^
        --images_dir "duong\\dan\\thu\\muc\\anh" ^
        --instance_prompt "1boy, silver hair with black and yellow streaks, red eyes" ^
        --steps 800

Chuẩn bị dữ liệu:
- 10–20 ảnh nhân vật (ảnh đã sinh + được duyệt, hoặc ảnh gốc), mặt rõ, đa dạng góc/pose.
- Đặt tất cả vào 1 thư mục. Không cần caption từng ảnh — dùng chung --instance_prompt.

Kết quả:
- File resource/character_loras/<slug>.safetensors — pipeline TỰ ĐỘNG load khi
  sinh ảnh có nhân vật này. Commit file vào git để máy khác pull về dùng ngay.

VRAM: ~5GB (512px, batch 1, gradient checkpointing) — chạy được trên RTX 3060 6GB.
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image

# Cho phép chạy từ bất kỳ CWD nào
_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC_ROOT)

OUTPUT_DIR = os.path.join(_MC_ROOT, "resource", "character_loras")
DEFAULT_CHECKPOINT = "stablediffusionapi/anything-v5"


def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA nhân vật (SD1.5, DreamBooth-style)")
    p.add_argument("--character", required=True, help="Slug nhân vật (VD: d_ch_phong)")
    p.add_argument("--images_dir", required=True, help="Thư mục chứa 10-20 ảnh nhân vật")
    p.add_argument("--instance_prompt", required=True,
                   help="Mô tả cố định cho nhân vật (tag tiếng Anh, VD: '1boy, silver hair, red eyes')")
    p.add_argument("--checkpoint", default="", help="Model SD gốc (mặc định: đọc từ context, fallback anything-v5)")
    p.add_argument("--steps", type=int, default=800, help="Số bước train (600-1200 là hợp lý)")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (8-32; cao hơn = giống hơn nhưng file to hơn)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_images(images_dir: str, resolution: int):
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))
             if f.lower().endswith(exts)]
    if len(paths) < 5:
        raise SystemExit(f"[LOI] Chi tim thay {len(paths)} anh trong {images_dir}. Can toi thieu 5 anh (khuyen nghi 10-20).")
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        # Resize giữ tỷ lệ rồi center-crop về vuông resolution
        w, h = img.size
        scale = resolution / min(w, h)
        img = img.resize((max(resolution, int(w * scale)), max(resolution, int(h * scale))), Image.LANCZOS)
        w, h = img.size
        left, top = (w - resolution) // 2, (h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))
        images.append(img)
    print(f"[INFO] Da nap {len(images)} anh tu {images_dir}")
    return images


def to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np
    arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # [-1, 1]
    return arr.permute(2, 0, 1)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[CANH BAO] Khong co GPU CUDA — train tren CPU se RAT cham (nhieu gio).")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    print(f"[INFO] Base model: {checkpoint}")

    cache_dir = os.path.join(_MC_ROOT, "storage", "models")

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        checkpoint, torch_dtype=torch.float32, safety_checker=None, cache_dir=cache_dir
    )
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    vae, unet = pipe.vae, pipe.unet
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze toàn bộ, chỉ train LoRA gắn vào UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    from peft import LoraConfig
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    unet.enable_gradient_checkpointing()

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in lora_params)
    print(f"[INFO] Trainable LoRA params: {n_params/1e6:.2f}M (rank {args.rank})")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Encode prompt một lần (dùng chung cho mọi ảnh)
    prompt = f"masterpiece, best quality, {args.instance_prompt}"
    with torch.no_grad():
        ids = tokenizer(prompt, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        encoder_hidden_states = text_encoder(ids)[0]

    # Encode toàn bộ ảnh sang latent trước (tiết kiệm VRAM khi train)
    images = load_images(args.images_dir, args.resolution)
    latents_list = []
    with torch.no_grad():
        for img in images:
            px = to_tensor(img).unsqueeze(0).to(device)
            latent = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            latents_list.append(latent.cpu())
    vae.to("cpu")
    torch.cuda.empty_cache() if device == "cuda" else None

    print(f"[INFO] Bat dau train {args.steps} buoc (grad_accum={args.grad_accum})...")
    unet.train()
    for step in range(1, args.steps + 1):
        latent = random.choice(latents_list).to(device)
        # Flip ngang ngẫu nhiên ở mức latent để tăng đa dạng
        if random.random() < 0.5:
            latent = torch.flip(latent, dims=[3])

        noise = torch.randn_like(latent)
        timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
        noisy_latent = noise_scheduler.add_noise(latent, noise, timestep)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            noise_pred = unet(noisy_latent, timestep, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float()) / args.grad_accum

        scaler.scale(loss).backward()

        if step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == 1:
            print(f"  step {step}/{args.steps} | loss {loss.item() * args.grad_accum:.4f}")

    # Lưu LoRA theo định dạng diffusers (load_lora_weights đọc được)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    unet_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    weight_name = f"{args.character}.safetensors"
    StableDiffusionPipeline.save_lora_weights(
        save_directory=OUTPUT_DIR,
        unet_lora_layers=unet_lora_state,
        weight_name=weight_name,
        safe_serialization=True,
    )
    out_path = os.path.join(OUTPUT_DIR, weight_name)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n[XONG] Da luu LoRA: {out_path} ({size_mb:.1f} MB)")
    print("[INFO] Pipeline se TU DONG dung LoRA nay khi sinh anh co nhan vat tren.")
    print("[INFO] Commit file vao git de may khac pull ve dung ngay:")
    print(f"       git add \"apps/MediaComposer/resource/character_loras/{weight_name}\"")


if __name__ == "__main__":
    main()
