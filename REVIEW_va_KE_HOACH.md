# BÁO CÁO REVIEW DỰ ÁN AIVOICE & KẾ HOẠCH KHẮC PHỤC

Ngày review: 05/07/2026. Phạm vi: toàn dự án, trọng tâm core sinh ảnh Workflow 5 (AI Storytelling) trong `apps/MediaComposer`.

---

## PHẦN A — NGUYÊN NHÂN GỐC: NHÂN VẬT KHÔNG ĐỒNG NHẤT

### A1. IP-Adapter FaceID KHÔNG BAO GIỜ được load thành công (lỗi nghiêm trọng nhất) 🔴

File: `app/services/storytelling/image_generator.py` (hàm `warmup`)

```python
self._pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None,
                           weight_name="ip-adapter-faceid_sd15.bin", cache_dir=cache_dir)
```

Thiếu tham số `image_encoder_folder=None`. Diffusers mặc định sẽ đi tìm thư mục `image_encoder` bên trong repo `h94/IP-Adapter-FaceID` — repo này **không có** thư mục đó (FaceID dùng InsightFace embedding, không dùng CLIP encoder). Kết quả: `load_ip_adapter` ném exception, bị nuốt bởi `except Exception` → chỉ log warning `Could not load IP-Adapter FaceID` → `_ip_adapter_loaded = False`.

Hệ quả dây chuyền: `_get_combined_embedding()` trả về `None` → **mọi ảnh được sinh ra hoàn toàn KHÔNG có điều kiện khuôn mặt**. Nhân vật chỉ được "giữ" bằng text keywords → mỗi seed ra một khuôn mặt khác nhau. Đây chính là lý do "đã tham chiếu cùng nhân vật mà ảnh vẫn không đồng nhất".

Cách sửa:
```python
self._pipe.load_ip_adapter(
    "h94/IP-Adapter-FaceID", subfolder=None,
    weight_name="ip-adapter-faceid_sd15.bin",
    image_encoder_folder=None,   # BẮT BUỘC với FaceID
    cache_dir=cache_dir)
```

### A2. Sai shape/format của face embedding — sẽ CRASH ngay khi A1 được sửa 🔴

File: `image_generator.py` (`_get_combined_embedding`)

Hiện tại truyền tensor `[1, 512]`. Diffusers yêu cầu:
- Shape `[batch, num_images, 512]` → phải là `[1, 1, 512]` (unsqueeze 2 lần).
- Khi `guidance_scale > 1` (CFG bật — UI mặc định 7.0), diffusers gọi `single_image_embeds.chunk(2)` để tách negative/positive → phải concat thêm **zero tensor làm negative**: `torch.cat([torch.zeros_like(emb), emb], dim=0)` → shape `[2, 1, 512]`. Thiếu phần này → crash `not enough values to unpack` với mọi ảnh có nhân vật.

Code chuẩn:
```python
emb = torch.from_numpy(face_embedding).to(self.device, dtype=self._pipe.unet.dtype)
emb = emb.reshape(1, 1, -1)                      # [1,1,512]
if guidance_scale > 1.0:
    emb = torch.cat([torch.zeros_like(emb), emb], dim=0)  # [2,1,512]
return [emb]
```
Lưu ý thêm: dtype đang hardcode `float16` → vỡ khi chạy CPU/fp32. Phải lấy dtype từ pipeline như trên.

### A3. Thiếu FaceID LoRA đi kèm 🟠

`ip-adapter-faceid_sd15.bin` được thiết kế để dùng **cùng** `ip-adapter-faceid_sd15_lora.safetensors` (cùng repo h94). Không fuse LoRA này (weight ~0.5–0.8) thì độ giống khuôn mặt giảm rõ rệt. Cần load + fuse, và quản lý adapter name để không xung đột với Hyper-SD LoRA.

### A4. StyleBuffer là code chết 🟠

- Với FaceID, `pipe.image_encoder` luôn `None` → `add_accepted_image()` luôn warning và return → buffer rỗng vĩnh viễn.
- Kể cả có, embedding CLIP (1024-dim) không bao giờ cùng shape với InsightFace (512-dim) → nhánh trộn `0.75*face + 0.25*style` không bao giờ chạy.
→ Xoá hoặc thiết kế lại (trộn 2 embedding khác không gian vector là sai về nguyên lý).

### A5. Mâu thuẫn giữa warmup và tham số generate 🟠

`warmup()` quyết định scheduler + có fuse Hyper-SD LoRA hay không dựa trên `num_inference_steps` trong **config.toml** (hiện = 25 → DPM++, không LoRA). Nhưng lúc sinh ảnh lại dùng steps/guidance từ **UI slider**. Nếu user chỉnh slider xuống 8 steps trong khi config 25 (hoặc ngược lại) → chạy sai chế độ: Hyper-SD LoRA + CFG 7 + 25 steps cho ảnh cháy/bệt; hoặc DPM 2 steps cho ảnh nhiễu. Cần truyền tham số thực tế vào warmup hoặc tách 2 profile rõ ràng.

### A6. Các yếu tố làm giảm đồng nhất/chất lượng khác 🟡

- **Độ phân giải 896×512, 1024×448 trên SD1.5**: model gốc train 512px; khung quá rộng gây lặp nhân vật, biến dạng anatomy. Nên sinh ở ~768 cạnh dài (768×432) rồi upscale, hoặc chuyển SDXL.
- **Tên nhân vật tiếng Việt `@DịchPhong` đưa thẳng vào prompt**: CLIP không hiểu, chỉ là token nhiễu. Identity nên để IP-Adapter lo; trong prompt chỉ giữ keywords ngoại hình.
- **Fast mode guidance ≤ 1.0** → negative prompt bị bỏ qua hoàn toàn (CFG tắt) mà UI vẫn hiển thị như có tác dụng.
- **`add_character()` reset `has_embedding = False`** khi cập nhật nhân vật (context_manager.py) dù file `face.ipadpt.npy` vẫn tồn tại → workflow video (orchestrator) bỏ qua khuôn mặt, trong khi Studio ảnh (image_gen_service) lại chỉ check file → 2 luồng hành xử khác nhau, dễ "lúc giống lúc không".
- **`quality_mode` unfuse/fuse LoRA bằng `try/except: pass`** → nếu unfuse fail giữa chừng, trạng thái pipeline hỏng âm thầm cho các lần sinh sau.
- **Seed batch**: `generate_batch` tạo generator đúng, nhưng orchestrator release() pipeline sau mỗi lần bấm → model load lại từ đầu mỗi batch (chậm) chứ không sai kết quả.

---

## PHẦN B — MÔI TRƯỜNG / GPU / THƯ VIỆN

### B1. GPU: RTX 3060 Laptop 6GB (Ampere, sm_86) ✅ — ĐÃ XÁC NHẬN VỚI CHỦ DỰ ÁN

- torch 2.6.0+cu124 hiện tại **tương thích đầy đủ** với sm_86 → không cần cài lại PyTorch.
- `GPU_SETUP.md` (viết cho RTX 5060/Blackwell) không áp dụng cho máy này — nên thêm ghi chú trong file để tránh nhầm lẫn sau này. `requirements.txt` ghi index cu121 trong khi venv là cu124: không gây lỗi, nhưng nên đồng bộ về cu124.
- **Hệ quả với 6GB VRAM**: `hardware_adapter.py` sẽ resolve về profile `cuda_low` (ngưỡng ≥7GB mới lên `cuda_high`) → SD chạy CPU offload, InsightFace + Whisper chạy CPU. Đây là hành vi đúng, nhưng có nghĩa:
  - Khi bật đủ SD fp16 + IP-Adapter FaceID + LoRA: ~4–5GB VRAM, vẫn vừa 6GB nếu offload hoạt động.
  - RealESRGAN nên hạ `tile_size` xuống 256 để tránh OOM khi upscale 4K.
  - Không nên chạy đồng thời Workflow 5 và các engine TTS GPU (XTTS) trong cùng phiên.

### B2. Trạng thái thư viện

| Thư viện | Phiên bản | Nhận xét |
|---|---|---|
| torch/torchvision | 2.6.0+cu124 / 0.21.0+cu124 | ✅ Tương thích RTX 3060 (sm_86) |
| diffusers | 0.38.0 | OK, hỗ trợ FaceID nếu gọi đúng API |
| basicsr | 1.4.2 (đã patch tay `functional_tensor`) | OK nhưng patch nằm trong venv — cài lại là mất. Cần ghi vào setup.bat |
| realesrgan | 0.3.0 + weights anime_6B, animevideov3 có sẵn | OK. Nhưng nếu chọn model `RealESRGAN_x4plus` (bản thường) code vẫn dựng RRDBNet `num_block=6` → sai kiến trúc (bản thường cần 23) → load fail |
| insightface | 1.0.1 + onnxruntime-gpu 1.20.2 | OK với Ampere; profile cuda_low đã ép InsightFace chạy CPU (đúng cho 6GB VRAM) |
| numpy | 2.4.6 | Rủi ro với fairseq/rvc (tuyên bố numpy<=1.23); hiện "chạy được" nhưng là nợ kỹ thuật |
| openai | 2.44.0 | OK |

### B3. Bảo mật 🔴

- **API key hardcode trong source**: `image_generator.py` chứa fallback `"sk-gemini-YrVwXWGegzkFlevHPdQy7Fpry14HJVirqvnuxukz"`; `config.toml` chứa key Pexels + OpenAI key thật và **không nằm trong .gitignore an toàn** nếu repo được share. → Xoá key khỏi code, chuyển sang biến môi trường / config không commit.

---

## PHẦN C — LỖI TIỀM ẨN KHÁC TRONG DỰ ÁN

1. **`get_llm_client()` nằm ngoài `try`** ở `prompt_translator._call_llm_json`, `character_extractor.call_llm`, `llm_prompter._call_llm` → chưa cấu hình API key là Streamlit crash nguyên trang thay vì báo lỗi đẹp.
2. **Đường dẫn tương đối** `storage/contexts`, `storage/tasks` (context_manager, orchestrator) → phụ thuộc CWD; chạy ngoài `run.bat` là hỏng. Nên dùng đường dẫn tuyệt đối từ `_MC_ROOT` như postprocess.py đã làm.
3. **`run_adetailer` là stub** — detect mặt xong chỉ log "Inpainting is simulated" rồi trả ảnh nguyên vẹn. Tính năng sửa mặt thực tế chưa tồn tại dù pipeline gọi nó.
4. **video_assembler bỏ qua config**: `video_fps=24` trong config nhưng ffmpeg hardcode `-r 25`; `video_codec = h264_nvenc` trong config bị bỏ qua (luôn libx264); style phụ đề chỉ áp FontName/FontSize, bỏ màu/viền/vị trí.
5. **Gemini API batch**: `images.generate(n=batch_size)` — nhiều proxy Gemini local không hỗ trợ `n>1`; đã có fallback tuần tự nhưng nên mặc định tuần tự.
6. **Singleton StorytellingPipeline + Streamlit**: mỗi rerun script tạo call `__init__` mới trên cùng instance; không khoá thread → 2 tab/2 phiên có thể đua nhau release/warmup.
7. **`step2_generate_images` (Studio) `release()` toàn bộ pipeline sau mỗi batch** → lần bấm kế tiếp load lại model từ disk (~30–60s). Nên giữ model, chỉ `empty_cache()`.
8. **`ImageOps.fit` sau upscale** crop cứng về 16:9 bất kể ảnh 9:16/1:1 → xuất ảnh dọc bị cắt nát khi target 1920×1080. Cần target theo aspect ratio ảnh gốc.
9. Warmup luôn tải model từ HuggingFace nếu cache thiếu — không có kiểm tra offline/thông báo tiến độ cho user trong UI.

---

## PHẦN D — KẾ HOẠCH THEO PHASE

### Phase 0 — Dọn môi trường (0.5 ngày) — GPU đã xác nhận OK (RTX 3060 6GB)
- ~~Cài lại PyTorch~~ Không cần: cu124 tương thích sm_86.
- Đồng bộ `requirements.txt` về index cu124; ghi chú vào `GPU_SETUP.md` rằng file chỉ dành cho RTX 50-series; ghi patch basicsr vào setup.bat để không mất khi cài lại venv.
- Xoá API key fallback hardcode trong `image_generator.py` (dù là key local, không nên nằm trong source); đưa `config.toml` vào .gitignore nếu repo có thể được chia sẻ. Key Pexels là free-tier, rủi ro thấp — nhưng nếu public repo thì người khác có thể xài hết quota của bạn.
- Tinh chỉnh cho 6GB VRAM: hạ RealESRGAN tile_size 512→256; giữ mặc định profile `cuda_low` (CPU offload).

### Phase 1 — Sửa core sinh ảnh: identity nhân vật (1–2 ngày) ← TRỌNG TÂM
1. Sửa `load_ip_adapter(..., image_encoder_folder=None)` (A1).
2. Sửa shape embedding `[1,1,512]` + zero-negative khi CFG (A2), dtype theo pipeline.
3. Load + fuse `ip-adapter-faceid_sd15_lora.safetensors` với adapter name riêng (A3).
4. Xoá/vô hiệu StyleBuffer (A4).
5. Đồng bộ warmup ↔ tham số generate: truyền steps/guidance thật vào warmup, tách rõ 2 profile "Fast (Hyper-SD, CFG≤1.2)" và "Quality (DPM++, CFG 5–7)" (A5).
6. Sửa `has_embedding` (không reset khi update nhân vật) và thống nhất logic check embedding giữa orchestrator & image_gen_service.
7. Log cảnh báo hiển thị LÊN UI khi IP-Adapter không load được hoặc scene không tìm thấy face embedding (không nuốt lỗi âm thầm nữa).
- Tiêu chí nghiệm thu: sinh 8 ảnh cùng 1 nhân vật, 8 seed khác nhau, cả 2 chế độ Fast/Quality — khuôn mặt nhận diện cùng identity (cosine similarity InsightFace > 0.5 giữa các ảnh).

### Phase 2 — Chất lượng ảnh (1–2 ngày)
- Giảm resolution sinh gốc về ≤768 cạnh dài theo aspect ratio, upscale bù bằng RealESRGAN.
- Bỏ `@TênNhânVật` khỏi prompt; chỉ giữ keywords ngoại hình tiếng Anh.
- Sửa RRDBNet num_block theo model (6 cho anime_6B, 23 cho x4plus); sửa `ImageOps.fit` crop theo aspect ratio ảnh.
- Triển khai ADetailer thật (inpaint mặt bằng chính pipeline SD + mask từ InsightFace) — thay stub.
- Cân nhắc nâng cấp: hỗ trợ SDXL + IP-Adapter FaceID SDXL, hoặc InstantID (giữ identity tốt hơn FaceID rõ rệt) làm lựa chọn trong UI.

### Phase 3 — Độ ổn định & UX (1–2 ngày)
- Bọc `get_llm_client()` vào try/except, hiện lỗi thân thiện trên Streamlit.
- Chuyển mọi đường dẫn `storage/...` sang tuyệt đối theo `_MC_ROOT`.
- Giữ pipeline trong VRAM giữa các batch (chỉ release khi đổi model / user bấm giải phóng).
- Thêm khoá (lock) quanh warmup/release để tránh race condition Streamlit.
- Áp dụng `video_fps`, `video_codec` (NVENC), full style phụ đề trong video_assembler.

### Phase 4 — Cải tiến mở rộng (tuỳ chọn, 3–5 ngày)
- **Consistency nâng cao**: LoRA nhân vật train nhanh (10–20 ảnh) cho nhân vật chính; hoặc ControlNet reference-only.
- **Batch thông minh**: sinh nhiều ảnh/lần bằng batched inference khi VRAM cho phép (thay vì tuần tự).
- **Semantic cache prompt LLM** để không dịch lại mô tả giống nhau.
- **Kiểm thử tự động**: test smoke cho từng workflow (đã có `chay_kiem_thu.bat` — mở rộng cover storytelling).
- **TTS side**: kiểm tra numpy 2.x với fairseq/RVC, cân nhắc thay fairseq bằng bản fork maintained.

---

## PHỤ LỤC — Danh sách file cần sửa (Phase 1)

| File | Nội dung sửa |
|---|---|
| `app/services/storytelling/image_generator.py` | A1, A2, A3, A4, A5, xoá API key hardcode |
| `app/services/storytelling/context_manager.py` | has_embedding, đường dẫn tuyệt đối |
| `app/services/storytelling/image_gen_service.py` | thống nhất check embedding, không release sau mỗi batch |
| `app/services/storytelling/orchestrator.py` | thống nhất check embedding, đường dẫn tuyệt đối |
| `app/services/storytelling/postprocess.py` | RRDBNet num_block, crop theo aspect, ADetailer thật |
| `app/services/llm.py` + các caller | error handling LLM |
| `requirements.txt`, `setup.bat`, `GPU_SETUP.md` | đồng bộ cu128, patch basicsr |
