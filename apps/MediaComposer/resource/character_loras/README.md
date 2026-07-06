# LoRA Nhân Vật

Thư mục chứa LoRA nhân vật cho AI Storytelling (Workflow 5).

- File `<slug>.safetensors` (VD: `d_ch_phong.safetensors`) sẽ được pipeline **tự động load** khi sinh ảnh có nhân vật tương ứng (trọng số 0.8).
- Train bằng: `train_lora.bat <slug> <thư_mục_ảnh> "<instance_prompt>" [số_bước]` (chạy trong thư mục MediaComposer).
- File LoRA rank 16 chỉ ~10–40MB → **commit thẳng vào git**, máy khác pull về là dùng ngay, không cần train lại.
- Slug nhân vật xem trong Context Window (VD: `d_ch_phong`, `l_c_lan_tuy_t`).
