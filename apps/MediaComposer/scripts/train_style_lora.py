# -*- coding: utf-8 -*-
"""
Train LoRA PHONG CÁCH cho AI Storytelling — khóa cả truyện vào một art direction.

Khác LoRA nhân vật ở ba điểm cốt lõi:

1. **Caption khác nhau cho từng ảnh.** LoRA nhân vật dùng chung một prompt vì ta
   MUỐN model gắn chặt vào một khuôn mặt. LoRA phong cách thì ngược lại: nếu mọi
   ảnh dùng chung caption, model sẽ học luôn NỘI DUNG (cùng một nhân vật, cùng
   một bối cảnh) chứ không tách được PHONG CÁCH. Vì vậy script này đọc caption
   riêng từ file .txt cùng tên ảnh, và bắt buộc mỗi caption phải mở đầu bằng
   trigger token chung (mặc định lấy từ --style_token).
2. **Dataset lớn hơn, đa dạng nội dung.** 40-80 ảnh, nhiều chủ thể/bối cảnh khác
   nhau nhưng CÙNG một cách vẽ. Ít ảnh mà cùng nội dung = LoRA nội dung, không
   phải LoRA phong cách.
3. **Rank thấp hơn, LR thấp hơn.** Phong cách là tín hiệu tần số thấp; rank 8
   là đủ và ít bám nội dung hơn rank 16-32.

Cách dùng (chạy từ thư mục apps/MediaComposer, dùng venv của dự án):
    ..\\..\\.venv\\Scripts\\python.exe scripts\\train_style_lora.py ^
        --style thuy_mac ^
        --images_dir "duong\\dan\\dataset" ^
        --style_token "thuymac ink wash style" ^
        --steps 1500

Chuẩn bị dữ liệu:
- 40-80 ảnh CÙNG phong cách, KHÁC nội dung (người/cảnh/vật, gần/xa, sáng/tối).
- Mỗi ảnh kèm 1 file .txt cùng tên mô tả NỘI DUNG ảnh đó bằng tag tiếng Anh.
  VD: anh01.png + anh01.txt chứa "a man walking on a stone bridge, mountains, fog".
  Thiếu .txt thì script tự dùng caption chung (--fallback_caption) và CẢNH BÁO —
  chất lượng tách phong cách sẽ kém hơn.

Kết quả:
- resource/style_loras/<style>.safetensors
- Bật bằng config.toml: style_lora = "<style>", style_lora_weight = 0.8

VRAM: ~3.5GB (512px, batch 1, gradient checkpointing, UNet fp16) — vừa RTX 3060 6GB.
Thời gian: ~35-50 phút cho 1500 bước trên RTX 3060 6GB.

Ghi chú về bộ nhớ (đã trả giá để học):
- UNet phải ở **fp16**. Nạp fp32 tốn ~3.4GB chỉ riêng trọng số, cộng VAE + text
  encoder + activation là chạm trần 6GB → driver tràn sang shared memory (RAM),
  RAM cạn theo, Windows paging xuống đĩa. Không crash, chỉ chậm đi hàng chục lần
  với GPU utilization ~30%. Tham số LoRA vẫn giữ fp32 để optimizer cập nhật ổn định.
- VAE và text encoder phải được **xoá hẳn** sau khi encode xong, không chỉ .to("cpu").
- Chỉ được chạy MỘT tiến trình train tại một thời điểm — script tự khoá bằng file.
"""
import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

# Cho phép chạy từ bất kỳ CWD nào
_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC_ROOT)

OUTPUT_DIR = os.path.join(_MC_ROOT, "resource", "style_loras")
DEFAULT_CHECKPOINT = "stablediffusionapi/anything-v5"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MIN_IMAGES = 15


def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA phong cách (SD1.5)")
    p.add_argument("--style", required=True,
                   help="Tên phong cách, thành tên file .safetensors (VD: thuy_mac)")
    p.add_argument("--images_dir", required=True,
                   help="Thư mục 40-80 ảnh cùng phong cách + file .txt caption cùng tên")
    p.add_argument("--style_token", default="",
                   help="Trigger token gắn đầu mọi caption (mặc định: '<style> style')")
    p.add_argument("--fallback_caption", default="an illustration",
                   help="Caption dùng khi ảnh thiếu file .txt")
    p.add_argument("--checkpoint", default="", help="Model SD gốc (mặc định anything-v5)")
    p.add_argument("--steps", type=int, default=1500,
                   help="Số bước train (1200-2500 hợp lý cho phong cách)")
    p.add_argument("--rank", type=int, default=8,
                   help="LoRA rank — phong cách nên để THẤP (8) để khỏi bám nội dung")
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(images_dir: str, resolution: int, style_token: str,
                 fallback_caption: str):
    """Trả list (PIL.Image vuông resolution, caption) — caption đọc từ .txt kèm ảnh."""
    if not os.path.isdir(images_dir):
        raise SystemExit(f"[LOI] Khong tim thay thu muc: {images_dir}")

    paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))
             if f.lower().endswith(IMAGE_EXTS)]
    if len(paths) < MIN_IMAGES:
        raise SystemExit(
            f"[LOI] Chi co {len(paths)} anh trong {images_dir}. "
            f"LoRA phong cach can toi thieu {MIN_IMAGES} anh (khuyen nghi 40-80), "
            f"NOI DUNG phai KHAC nhau nhung cach ve GIONG nhau.")

    samples = []
    missing_caption = 0
    for path in paths:
        img = Image.open(path).convert("RGB")
        # Resize giữ tỷ lệ rồi center-crop về vuông resolution
        w, h = img.size
        scale = resolution / min(w, h)
        img = img.resize((max(resolution, int(w * scale)),
                          max(resolution, int(h * scale))), Image.LANCZOS)
        w, h = img.size
        left, top = (w - resolution) // 2, (h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))

        caption_path = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            caption = fallback_caption
            missing_caption += 1
        samples.append((img, f"{style_token}, {caption}".strip(", ")))

    print(f"[INFO] Da nap {len(samples)} anh tu {images_dir}")
    if missing_caption:
        print(f"[CANH BAO] {missing_caption}/{len(samples)} anh THIEU file .txt caption. "
              f"Dung caption chung '{fallback_caption}' — LoRA se de bam noi dung "
              f"thay vi chi hoc phong cach. Nen viet caption cho tung anh.")
    return samples


def to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np
    arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # [-1, 1]
    return arr.permute(2, 0, 1)


class SingleRunLock:
    """Khoá file chống chạy hai tiến trình train cùng lúc.

    Hai trainer trên cùng một GPU 6GB không báo lỗi — chúng cùng tràn sang shared
    memory rồi kéo nhau xuống paging. Triệu chứng nhìn giống 'máy yếu' nên rất dễ
    chẩn đoán nhầm. Chặn ngay từ đầu rẻ hơn nhiều so với đi tìm nguyên nhân sau.
    """

    def __init__(self, path: str):
        self.path = path
        self.acquired = False

    def __enter__(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    pid = int(f.read().strip())
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and _pid_alive(pid):
                raise SystemExit(
                    f"[LOI] Da co tien trinh train dang chay (PID {pid}).\n"
                    f"      Dung no truoc, hoac xoa {self.path} neu chac chan da chet.")
            print(f"[INFO] Don khoa cu cua PID {pid} da chet.")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        self.acquired = True
        return self

    def __exit__(self, *exc):
        if self.acquired and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        return False


def _pid_alive(pid: int) -> bool:
    try:
        import subprocess
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=15).stdout
        return str(pid) in out
    except Exception:
        return False


def free_memory():
    import gc
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def report_vram(tag: str):
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    print(f"[VRAM {tag}] reserved={reserved:.2f}GB / {total:.2f}GB "
          f"({reserved / total * 100:.0f}%)")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[CANH BAO] Khong co GPU CUDA — train tren CPU se RAT cham (nhieu gio).")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    style_token = args.style_token or f"{args.style.replace('_', ' ')} style"
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    print(f"[INFO] Base model: {checkpoint}")
    print(f"[INFO] Trigger token: '{style_token}'")

    cache_dir = os.path.join(_MC_ROOT, "storage", "models")

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        checkpoint, torch_dtype=torch.float32, safety_checker=None, cache_dir=cache_dir
    )
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    vae, unet = pipe.vae, pipe.unet
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    samples = load_dataset(args.images_dir, args.resolution, style_token,
                           args.fallback_caption)
    n_images = len(samples)

    # --- Pha 1: encode latent + caption, CHỈ có VAE và text encoder trên GPU ----
    # Mỗi ảnh một caption riêng (điểm khác cốt lõi so với LoRA nhân vật).
    print("[INFO] Pha 1/2: encode latent + caption...")
    vae.to(device)
    text_encoder.to(device)
    report_vram("pha 1")

    pairs = []
    with torch.no_grad():
        for img, caption in samples:
            px = to_tensor(img).unsqueeze(0).to(device)
            latent = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            ids = tokenizer(caption, padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length,
                            return_tensors="pt").input_ids.to(device)
            emb = text_encoder(ids)[0]
            # Giữ trên CPU ở fp16: 46 ảnh chỉ tốn vài MB, khỏi chiếm VRAM.
            pairs.append((latent.half().cpu(), emb.half().cpu()))

    # XOÁ HẲN VAE + text encoder. Chỉ .to("cpu") thôi thì pipe vẫn giữ tham chiếu
    # và bộ nhớ đệm CUDA của chúng không được trả lại.
    del vae, text_encoder, samples
    pipe.vae = None
    pipe.text_encoder = None
    del pipe
    free_memory()
    report_vram("sau khi giai phong VAE/TE")

    # --- Pha 2: chỉ UNet trên GPU, ở fp16 ------------------------------------
    from peft import LoraConfig
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    unet.enable_gradient_checkpointing()

    # Trọng số đóng băng ở fp16 (~1.7GB thay vì ~3.4GB); riêng tham số LoRA phải
    # ép lại fp32, nếu để fp16 thì bước cập nhật của AdamW mất chính xác và loss
    # sẽ đứng yên.
    unet.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
    if device == "cuda":
        try:
            from diffusers.training_utils import cast_training_params
            cast_training_params(unet, dtype=torch.float32)
        except Exception:
            for p in unet.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
    free_memory()
    report_vram("pha 2 (UNet fp16)")

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in lora_params)
    print(f"[INFO] Trainable LoRA params: {n_params/1e6:.2f}M (rank {args.rank})")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"[INFO] Bat dau train {args.steps} buoc (grad_accum={args.grad_accum})...")
    unet.train()
    t_start = time.time()
    for step in range(1, args.steps + 1):
        latent_cpu, emb_cpu = random.choice(pairs)
        dtype = torch.float16 if device == "cuda" else torch.float32
        latent = latent_cpu.to(device=device, dtype=dtype)
        encoder_hidden_states = emb_cpu.to(device=device, dtype=dtype)
        if random.random() < 0.5:
            latent = torch.flip(latent, dims=[3])

        noise = torch.randn_like(latent)
        timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                 (1,), device=device)
        noisy_latent = noise_scheduler.add_noise(latent, noise, timestep)

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device == "cuda")):
            noise_pred = unet(noisy_latent, timestep,
                              encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float()) / args.grad_accum

        scaler.scale(loss).backward()

        if step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            rate = step / max(elapsed, 1e-6)
            eta = (args.steps - step) / max(rate, 1e-6)
            print(f"  step {step}/{args.steps} | loss {loss.item() * args.grad_accum:.4f} "
                  f"| {rate:.1f} b/s | con ~{eta/60:.1f} phut", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    unet_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    weight_name = f"{args.style}.safetensors"
    StableDiffusionPipeline.save_lora_weights(
        save_directory=OUTPUT_DIR,
        unet_lora_layers=unet_lora_state,
        weight_name=weight_name,
        safe_serialization=True,
    )
    out_path = os.path.join(OUTPUT_DIR, weight_name)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    import json
    with open(os.path.join(OUTPUT_DIR, f"{args.style}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": checkpoint,
            "style_token": style_token,
            "steps": args.steps,
            "rank": args.rank,
            "lr": args.lr,
            "n_images": n_images,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[XONG] Da luu LoRA phong cach: {out_path} ({size_mb:.1f} MB)")
    print("[INFO] Bat trong config.toml phan [storytelling]:")
    print(f'       style_lora = "{args.style}"')
    print('       style_lora_weight = 0.8')
    print(f"[INFO] Nho them '{style_token}' vao dau file style cua truyen "
          f"(resource/image_presets/*.txt) de kich hoat trigger token.")


if __name__ == "__main__":
    _lock_path = os.path.join(_MC_ROOT, "storage", "style_loras.lock")
    with SingleRunLock(_lock_path):
        main()
