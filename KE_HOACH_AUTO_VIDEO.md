# KẾ HOẠCH: TỰ ĐỘNG SINH VIDEO TRUYỆN HÀNG LOẠT (MediaComposer — Workflow 5)

> Tài liệu thiết kế chi tiết để một Agent khác thực hiện. Đọc kỹ mục "Bối cảnh & ràng buộc"
> trước khi viết bất kỳ dòng code nào. Làm theo đúng thứ tự Phase A → E.
> Ngày lập: 05/07/2026. Người duyệt yêu cầu: Duy (chủ dự án).

---

## 0. BỐI CẢNH & RÀNG BUỘC (BẮT BUỘC ĐỌC)

### 0.1. Hiện trạng đã có (KHÔNG làm lại)
- Core sinh ảnh đã ổn định sau đợt sửa 07/2026:
  - `app/services/storytelling/image_generator.py`: singleton `StorytellingPipeline`, IP-Adapter Plus Face (CLIP, engine `clip` — hỗ trợ anime) + FaceID (engine `faceid`), Hyper-SD/DPM++ theo steps, compel cho prompt >77 token, `set_character_lora(slug)` đã có sẵn — tự load `resource/character_loras/<slug>.safetensors` (weight 0.8).
  - `scripts/train_character_lora.py` + `train_lora.bat`: train LoRA SD1.5 từ 1 thư mục ảnh (CLI, ~5GB VRAM, 512px).
  - `context_manager.py`: `get_ref_image_path()`, `has_identity()`, `get_face_embedding_path()`.
  - `image_gen_service.py` (Studio ảnh) và `orchestrator.py` (video) đều truyền `face_image` + gọi `set_character_lora`.
- Tách cảnh hiện tại: `md_parser.py` chia theo đoạn văn + thời lượng ước lượng ~5s, `srt_mapper.py` ghép nhóm SRT theo khoảng lặng >0.5s rồi map tuyến tính 1-cảnh-1-nhóm. **Sẽ được thay thế ở Phase C** (giữ làm fallback).

### 0.2. Ràng buộc kỹ thuật
- Máy chính: RTX 3060 Laptop 6GB VRAM (profile `cuda_low`, CPU offload). Máy phụ: RTX 5060 8GB (cu128). KHÔNG được giả định >6GB VRAM.
- **Train LoRA và sinh ảnh KHÔNG chạy đồng thời** (tranh VRAM). Trước khi train phải gọi `StorytellingPipeline().release()`.
- Python 3.11, venv tại gốc dự án, chạy UI qua `apps/MediaComposer/run.bat` (Streamlit, port 8502).
- LLM: OpenAI-compatible local Gemini proxy (`config.toml` → `openai_base_url`), có thể fail bất kỳ lúc nào → mọi tính năng LLM PHẢI có fallback không-LLM.
- `context.json` cũ đang tồn tại trên máy user → mọi field mới trong dataclass `Character`/`StoryContext` PHẢI có giá trị mặc định, và loader phải bỏ qua key lạ (hiện `Character(**c)` sẽ crash với key lạ — phải sửa loader thành lọc key theo dataclass fields).
- Sau MỖI file sửa: chạy `py_compile`. Sau mỗi Phase: chạy smoke test của Phase đó (mô tả ở từng Phase).
- Quy ước đường dẫn: luôn tuyệt đối từ `_MC_ROOT` (xem các file đã sửa làm mẫu). KHÔNG dùng đường dẫn tương đối theo CWD.

### 0.3. Quyết định thiết kế đã chốt với chủ dự án
1. **Train LoRA**: KHÔNG train tự động khi tạo nhân vật. Trước khi bắt đầu một phiên sinh hàng loạt, hệ thống quét các nhân vật xuất hiện trong batch mà chưa có LoRA → hiện danh sách cho user tick chọn "train trước khi sinh". Nhân vật không được chọn (nhân vật phụ) → dùng IP-Adapter như hiện tại, hoặc không identity nếu thiếu ảnh ref. Không bao giờ chặn batch vì thiếu LoRA.
2. **Dataset ảnh train**: nguồn kép — (a) ảnh user upload khi tạo nhân vật + ảnh user bấm "Chấp nhận", luôn được nhận; (b) khi tổng < 15 ảnh, bổ sung ảnh sinh ra được lọc tự động bằng CLIP similarity (đánh dấu riêng, xoá được).
3. **Tách cảnh**: LLM đọc kịch bản chia cảnh theo ngữ cảnh. SRT chỉ dùng lấy timing.
4. **Nhịp cảnh**: 8–15 giây/cảnh linh hoạt (LLM quyết định trong khoảng, có post-rule ép biên).
5. **(Bổ sung 07/2026) Face detailer ĐÃ BỊ LOẠI** khỏi phạm vi (thử nghiệm kém hiệu quả, đã roll back — InsightFace không detect mặt anime). KHÔNG phase nào được phụ thuộc vào detailer. Chất lượng mặt dựa vào: tinh chỉnh tham số sinh (A0) + cổng chất lượng + LoRA.
6. **(Bổ sung 07/2026) Thu thập dataset từ batch VẪN BẬT** (quyết định của chủ dự án): cổng chất lượng sau upscale là bộ lọc duy nhất, không cần tương tác user.
7. **(Bổ sung 07/2026) Bootstrap ref tự động**: nhân vật xuất hiện trong batch mà CHƯA có ảnh ref trong Context Window (viết tắt: CW) → lấy ảnh ĐẠT CỔNG CHẤT LƯỢNG ĐẦU TIÊN sinh ra có nhân vật đó làm `ref.*` tự động, các cảnh sau của nhân vật đó dùng ref này qua IP-Adapter → đồng nhất từ ảnh thứ 2 trở đi. Chi tiết ở A5.

---

## PHASE M — HỢP NHẤT 3 CHẾ ĐỘ TỰ ĐỘNG THÀNH MỘT (lập 06/07/2026, CHƯA code)

### Hiện trạng: 3 chế độ chồng chéo trong Tab 5

| Chế độ | Luồng | Ưu điểm | Nhược điểm |
|---|---|---|---|
| ⚡ Chạy Tự Động Toàn Bộ | Upload 1 md + 1 audio (+srt) → `orchestrator.run_pipeline()` (luồng CŨ) | Tách cảnh dày (~5s/cảnh, md_parser), timing Whisper chuẩn → video nhiều phân cảnh, ổn định | 1 truyện/lần, không resume, không report, không semantic |
| 📦 Chạy Hàng Loạt (cũ) | Quét thư mục input, ghép cặp theo tên → luồng cũ | Nhiều truyện | Không state/resume/report, trùng chức năng |
| 📦🎬 Sinh Video Hàng Loạt (mới) | `batch_video_runner` + semantic split + pre-flight LoRA | Resume, BatchReport, bootstrap ref, shot-type | Tách cảnh ĐANG LỖI (xem M1) → video 2 cảnh |

### M1 — Vá nền tảng timing/duration (nguyên nhân semantic chỉ ra 2 cảnh) 🔴
Chuỗi lỗi từ log 06/07 12:48: pydub không tìm thấy ffmpeg → đọc duration audio fail → fallback **60s âm thầm** → semantic chia 8-15s/cảnh trên 60s → sau merge còn 2 cảnh; đồng thời nhánh semantic **không gọi Whisper** khi thiếu SRT (lỗ hổng #5 đã cảnh báo khi review plan cũ) → timing chia đều theo từ.
1. Trỏ pydub dùng ffmpeg của `imageio-ffmpeg` (đã có sẵn trong dự án): `AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()` tại điểm khởi động; với .wav đọc bằng module `wave` trước (không cần ffmpeg).
2. CẤM fallback 60s âm thầm: đọc duration fail → raise lỗi rõ ràng, item đó `failed` trong BatchReport với thông báo dễ hiểu.
3. (CHỐT 06/07 với chủ dự án) KHÔNG dùng Whisper: có SRT → dùng timing SRT; không có → TỰ TẠO SRT TỪ KỊCH BẢN (audio đọc nguyên văn kịch bản): text = kịch bản đã cắt, timing tỷ lệ số từ trên thời lượng thật (vá ở mục 1-2). Cải tiến nhỏ so với nhánh No-Whisper hiện có: chia block SRT theo CÂU (không phải theo cảnh — cảnh 15s làm 1 block phụ đề là quá dài để đọc), mỗi câu nhận timing tỷ lệ từ bên trong cảnh của nó.
Nghiệm thu M1: 1 file 10 phút KHÔNG có SRT → số cảnh ≈ duration/11s ±30%; video dài đúng bằng audio.

### M2 — Một lõi thực thi duy nhất
- `batch_video_runner.run_batch(items=[...])` là LÕI DUY NHẤT; trường hợp 1 truyện = list 1 item. 
- ⚡ mode: bỏ gọi `orchestrator.run_pipeline` trực tiếp — UI đóng gói file upload thành 1 BatchItem rồi gọi run_batch (được luôn resume + report + pre-flight LoRA + bootstrap cho cả chế độ đơn).
- TRƯỚC KHI XOÁ `run_pipeline`: đọc kỹ nó để không mất option riêng (bgm_path, bgm_volume, burn_subtitles, enable_upscaling, learned_corrections...) — mọi option phải có mặt trong `options` của run_batch và UI.
- 📦 cũ: xoá radio, chức năng gộp vào 📦🎬 (giữ quy ước ghép cặp theo tên đã quen dùng).

### M3 — UI gộp còn MỘT chế độ tự động
Radio Tab 5 còn 4 lựa chọn: `🛑 3 Trạm Tương Tác` | `🚀 Sinh Video Tự Động` | `🖼️ Sinh 1 Ảnh` | `🖼️📦 Sinh Ảnh Hàng Loạt`.
- Trong `🚀 Sinh Video Tự Động` có 2 tab con: **"1 Truyện"** (upload md+audio+srt tuỳ chọn, như ⚡ cũ) và **"Cả Thư Mục"** (nhập path như 📦🎬). Cùng khối Options: upscale, BGM, burn subtitle, chất lượng. KHÔNG có toggle tách cảnh — LLM semantic là mặc định duy nhất (fallback parser cũ tự động khi LLM lỗi, có ghi rõ trong report).
- Pre-flight LoRA hiển thị chung sau khi có danh sách item. Progress 2 tầng + BatchReport cuối (cả chế độ 1 truyện).

### M4 — Chính sách tách cảnh & SRT (ĐÃ CHỐT với chủ dự án 06/07)
- **Tách cảnh: LUÔN dùng LLM ngữ nghĩa** (semantic split) — đây là hành vi mặc định và duy nhất của chế độ tự động. Parser cũ CHỈ là fallback tự động khi LLM fail (không phải lựa chọn ngang hàng, không cần đối chứng chọn mặc định).
- **SRT hoàn toàn tuỳ chọn, KHÔNG dùng Whisper**: có SRT → dùng timing SRT; không có → **tự tạo SRT từ chính kịch bản** (audio là giọng đọc nguyên văn kịch bản nên text phụ đề = text cảnh đã cắt, timing chia theo tỷ lệ số từ trên thời lượng audio thật). SRT sinh ra dùng luôn cho burn subtitle — video không SRT đầu vào vẫn có phụ đề đầy đủ. (Nhánh này đã tồn tại trong `srt_mapper` — "No-Whisper mode" tự ghi SRT từ scenes; chỉ hỏng vì duration 60s, vá ở mục 1-2 là chạy đúng.)
- Kiểm chứng sau M1: chạy 1 file thật 2 lần (có SRT / không SRT) — số cảnh và ranh giới cảnh phải GIỐNG NHAU, chỉ start/end time khác nhẹ.

### M5 — Nghiệm thu tổng
- 1 truyện qua UI mới = kết quả ≥ chế độ ⚡ cũ (số cảnh, timing, video dài đúng audio).
- Batch 2 item + giết app giữa chừng → resume đúng.
- 3 Trạm Tương Tác và 2 chế độ Studio ảnh KHÔNG thay đổi hành vi.
- Dọn dẹp: gỡ stub `run_adetailer` cũ trong `postprocess.process_all` (đang load InsightFace vô ích ~5s/video, "Inpainting is simulated").

---

## ⚡ TRẠNG THÁI TRIỂN KHAI (rà soát code thật 06/07/2026)

| Phase | Trạng thái trong code | Ghi chú |
|---|---|---|
| A (dataset) | ✅ Đã làm, NHƯNG hook đặt SAI CHỖ | `maybe_collect` đang móc ở Trạm 2 (ảnh draft) trong `image_gen_service.step2` + `orchestrator.step2` + nút Chấp nhận — TRÁI với quyết định chống nhiễm "chỉ thu sau upscale". Chưa có gate kích thước mặt. → Sửa ở T6 |
| A5 (bootstrap ref) | ❌ CHƯA làm | Không có `ref_source` / logic bootstrap trong code → T7 |
| B (LoRA lifecycle) | ✅ `lora_trainer.py` + pre-flight | Chưa kiểm chứng end-to-end với train thật |
| C (tách cảnh LLM) | ✅ `semantic_scene_splitter.py` (354 dòng) + `srt_mapper` mở rộng | Cần nghiệm thu với data thật |
| D (batch runner) | ✅ `batch_video_runner.py` (384 dòng) | Cần nghiệm thu resume |
| Face Detailer | ✅ ĐÃ LÀM LẠI (v2, 06/07 01:06) | Bản v2 khác hẳn bản bị roll back: detect bằng lbpcascade_animeface chạy TRONG SUBPROCESS (fix crash OpenMP), img2img chia sẻ component, tự chuyển DPM++ 22 steps riêng cho mặt, CÓ cổng kiểm định (mặt mới phải detect được + độ giống ≥ mặt cũ) → hỏng thì giữ mặt gốc. Đang bật mặc định ở Trạm 2. Cần nghiệm thu T4 |

---

## PHASE T — TINH CHỈNH SINH ẢNH (ĐIỂM ĐAU HIỆN TẠI — làm TRƯỚC mọi thứ khác)

Chẩn đoán từ code thật, mỗi mục là 1 nguyên nhân → 1 fix cụ thể:

### T1. Ảnh MỜ — nguyên nhân số 1: upscaler đang TẮT 🔴
`config.toml` hiện có `enable_upscaling = false` → Trạm 3 chỉ resize PIL LANCZOS 768×432 → 1920×1080 (phóng 2.5× không AI) = mờ toàn tập. Weights RealESRGAN anime_6B ĐÃ có sẵn trong `models/realesrgan/`.
- Fix: đặt `enable_upscaling = true`. Đo thời gian thêm/ảnh (kỳ vọng ~2-4s với tile 256). Nếu chê chậm: thử `realesr-animevideov3` (nhẹ hơn ~3x, mềm hơn chút).
- Nghiệm thu: cùng 1 seed, so ảnh export trước/sau khi bật — chữ ký độ nét phải thấy rõ bằng mắt.

### T2. Studio và Batch đang chạy 2 bộ tham số KHÁC NHAU 🔴
`config.toml` = steps 8/guidance 5 (batch video dùng), nhưng slider Studio hardcode default **25/7.0** (Main.py 4 chỗ: dòng ~1817, ~1821, ~1962, ~1966). Hậu quả: "tinh chỉnh chế độ sinh đơn để tin batch" là VÔ NGHĨA khi 2 chế độ không cùng tham số.
- Fix: slider `value=` đọc từ `load_storytelling_config()` (steps, guidance, ip_scale) thay vì hardcode → config.toml là nguồn sự thật duy nhất; chỉnh slider xong có nút "💾 Lưu làm mặc định" ghi ngược vào config.
- Nghiệm thu: đổi config → mở UI thấy slider khớp; sinh đơn và batch cùng tham số cho ảnh cùng chất lượng (cùng seed → gần như cùng ảnh).

### T3. Chi tiết hành động (cầm ô, đứng giữa phố) bị "quên" 🟠
Hai nguyên nhân: (a) trọng số kiểu `(tag:1.3)` chỉ có tác dụng khi prompt đi qua compel, mà code hiện chỉ dùng compel khi >77 token → prompt ngắn thì `(x:1.2)` thành **token rác** đưa thẳng vào CLIP; (b) system prompt LLM chưa yêu cầu nhấn trọng số hành động.
- Fix 1: `_build_prompt_kwargs` LUÔN encode qua compel (bỏ điều kiện >77), kèm regex chuyển cú pháp A1111 `(tag:1.3)` → cú pháp compel `(tag)1.3`. Fallback prompt thường (đã strip weight) nếu compel thiếu/lỗi.
- Fix 2: sửa `TRANSLATE_SYSTEM_PROMPT` + `llm_prompter`: hành động chính đặt NGAY SAU style tags và bọc trọng số `(action:1.3)`; ví dụ mẫu trong prompt phải minh hoạ đúng.
- Nghiệm thu: mô tả chuẩn "Dịch Phong đứng giữa phố đêm mưa, tay phải cầm ô" — 4/5 ảnh phải có ô đúng tay + người đứng giữa khung + mưa.

### T4. Nghiệm thu Face Detailer v2 (mặt nhỏ/ngũ quan lệch) 🟠
Bản v2 đã có cổng kiểm định tự vứt mặt vẽ hỏng — nhưng chưa được nghiệm thu có hệ thống.
- Quét: `face_detailer_strength` 0.35 / 0.45 / 0.55 (config có sẵn key), cùng 10 ảnh chuẩn.
- Đọc log tỷ lệ: mặt được vẽ lại / bị gate vứt / bỏ qua (quá to, quá bé). Gate vứt >50% → nới dung sai similarity (0.02 → 0.05) hoặc tăng steps mặt 22→28.
- Nghiệm thu: 10 ảnh 16:9 có nhân vật, ≥7/10 ngũ quan thẳng, đúng nhân vật sau detailer + upscale.

### T5. Cảnh cận/trung sinh đúng khung — mặt to ngay từ draft 🟡
16:9 (768×432) cho MỌI cảnh khiến mặt ~50px ở cảnh lẽ ra là cận cảnh. LLM prompter (Phase C đã có metadata cảnh) bổ sung field `shot_type: close|medium|wide`:
- close → sinh 576×704 (dọc, mặt to), medium → 704×528, wide → 768×432. Sau upscale, video_assembler đã pad về 16:9 sẵn (`force_original_aspect_ratio=decrease,pad`) nên KHÔNG vỡ khung video — cảnh cận sẽ có viền pad 2 bên (chấp nhận, hoặc dùng blur-pad sau).
- Fallback không có shot_type → wide như cũ.
- Nghiệm thu: cảnh hội thoại cận mặt cho mặt ≥180px trước upscale.

### T6. Di chuyển hook dataset về ĐÚNG chỗ (chống nhiễm — code đang làm SAI quyết định) 🔴
Code hiện thu ảnh draft ở Trạm 2. Sửa theo quyết định đã chốt:
- Gỡ `maybe_collect` khỏi `image_gen_service.step2` + `orchestrator.step2` (auto) và nút Chấp nhận Trạm 2 (approved).
- Móc lại ở: `image_gen_service.step3_upscale_export` (approved — sau upscale) và post-process của `batch_video_runner` (auto — sau upscale).
- Thêm gate kích thước mặt: dùng `face_detailer.detect_faces()` sẵn có (subprocess, an toàn) — mặt cạnh ngắn ≥160px mới nhận. Giữ gate CLIP similarity hiện có (0.60; ngưỡng chuẩn sẽ chốt lại bằng số đo từ T4).
- Lưu dạng crop mặt 512×512 (bbox ×1.6 pad vuông — tái dùng `_expand_box` của face_detailer) + bản full khung khi mặt >25% khung.

### T7. Bootstrap ref tự động (A5 — CHƯA có trong code)
Giữ nguyên đặc tả A5 bên dưới. Điều kiện tiên quyết: T4 + T6 xong (vì ref bootstrap phải qua cùng cổng chất lượng).

**Thứ tự làm: T1 → T2 (2 fix nhanh, làm ngay cùng lúc) → T3 → T4 (phiên tinh chỉnh cùng chủ dự án) → T5 → T6 → T7.**
Chuẩn nghiệm thu tổng của Phase T: 10 ảnh liên tiếp cùng nhân vật qua full pipeline (draft → detailer → upscale) — **≥7/10 đạt** (mặt đúng nhân vật, ngũ quan thẳng, nét, đúng hành động). Đạt rồi mới bật thu thập dataset trong batch (T6) và bootstrap (T7).

---

## PHASE A — DATASET NHÂN VẬT TRONG CONTEXT WINDOW

Mục tiêu: mỗi nhân vật có một thư mục dataset ảnh chuẩn hoá, nạp từ 2 nguồn (upload + tự thu thập), làm nguyên liệu train LoRA.

### A1. Data model
File: `app/services/storytelling/models.py` — thêm field vào dataclass `Character` (tất cả có default):
```python
lora_status: str = "none"        # none | queued | training | trained | failed
lora_trained_at: str = ""        # ISO datetime
instance_prompt: str = ""        # tag train LoRA; rỗng = tự sinh từ keywords_en
auto_collect: bool = True        # cho phép tự thu thập ảnh từ quá trình sinh
```
File: `context_manager.py`:
- Sửa `load_context()`: lọc dict theo `dataclasses.fields(Character)` trước khi `Character(**c)` (chống crash key lạ, tương thích context cũ).
- Thêm helper:
  - `get_dataset_dir(slug)` → `storage/contexts/<story>/characters/<slug>/dataset/` (tạo nếu chưa có).
  - `count_dataset_images(slug)` → int (đếm `approved_*.png` + `auto_*.png`).
  - `add_dataset_image(slug, image_path_or_pil, source: "approved"|"auto")` → copy/save vào dataset với tên `approved_<8hex>.png` / `auto_<8hex>.png`; áp trần: tối đa 40 ảnh, khi vượt thì xoá `auto_*` cũ nhất trước (FIFO), KHÔNG bao giờ tự xoá `approved_*`.

### A2. UI Context Window (webui/Main.py, khu vực ~dòng 1000–1200)
- Form tạo/sửa nhân vật: đổi uploader ảnh ref thành `st.file_uploader(..., accept_multiple_files=True)`. Ảnh đầu tiên = `ref.*` (giữ hành vi cũ), TẤT CẢ ảnh upload được lưu thêm vào dataset dạng `approved_*`.
- Mỗi nhân vật hiển thị thêm: số ảnh dataset (`count_dataset_images`), trạng thái LoRA (badge theo `lora_status`), checkbox `auto_collect`, nút "🗑 Xoá ảnh auto" (xoá toàn bộ `auto_*.png`).
- KHÔNG thêm nút train ở đây (train nằm ở Phase B, gắn với phiên batch).

### A3. Thu thập ảnh tự động từ quá trình sinh
File mới: `app/services/storytelling/dataset_collector.py`
```python
def maybe_collect(ctx_mgr, char_slug, image: PIL.Image, source: str) -> bool
```
**⚠️ NGUYÊN TẮC CHỐNG NHIỄM DỮ LIỆU (cập nhật 07/2026 — KHÔNG còn face detailer):**
Ảnh draft ở Trạm 2 (mặt nhỏ ~50px, ngũ quan dễ lệch) KHÔNG đủ chuẩn train.
Thu thập dataset chỉ diễn ra SAU bước upscale (Trạm 3 / post-process của batch),
và MỌI ảnh (kể cả approved) phải qua cổng chất lượng. "Chấp nhận để làm video"
và "đủ chuẩn để train" là 2 tiêu chí khác nhau. Không có detailer nên cổng
chất lượng + bộ tham số chuẩn từ A0 là 2 lớp bảo vệ duy nhất — tuân thủ nghiêm.

Logic:
1. Điều kiện tiên quyết: `char_slug` hợp lệ, `character.auto_collect == True`, ảnh chỉ chứa 1 nhân vật (caller đảm bảo — chỉ gọi khi `len(task.character_slugs) == 1`).
2. **Cổng chất lượng (áp cho CẢ approved lẫn auto, trừ ảnh user upload tay):**
   - Ảnh phải là bản SAU upscale (Trạm 3 / post-process batch), không phải draft Trạm 2.
   - Vùng mặt detect được phải ≥ 160px cạnh ngắn. Detector mặt anime: dùng model YOLOv8 anime-face (tải 1 lần ~50MB, chạy CPU được) — KHÔNG dùng InsightFace (không detect mặt anime). Nếu detector chưa tải được → chỉ nhận ảnh khi mặt ước lượng theo shot_type là close/medium.
   - CLIP similarity với `ref.*` ≥ ngưỡng (approved: 0.55, auto: 0.65 — auto khắt khe hơn). Encoder lấy từ pipeline đang load (`pipe.image_encoder` + `pipe.feature_extractor`, nhớ cast đúng `encoder.dtype`); encoder chưa load → bỏ qua, không tự load model.
   - Cache ref embedding vào `dataset/.ref_emb.npy`, invalidate khi mtime của `ref.*` đổi.
3. **Lưu dạng crop mặt**: thay vì cả khung 16:9, crop vùng mặt (mở rộng 1.6× bbox, pad vuông) resize 512×512 — nguyên liệu lý tưởng cho LoRA train ở 512px. Lưu thêm bản khung đầy đủ chỉ khi mặt chiếm >25% khung (ảnh cận cảnh).
4. `source == "auto"`: chỉ nhận khi `count_dataset_images < 15`. `source == "approved"`: luôn xét (không giới hạn 15) nhưng vẫn qua cổng chất lượng.
5. Ảnh user upload tay khi tạo nhân vật: vào thẳng dataset (nguồn tin cậy, không qua cổng).
6. Ghi log mỗi lần nhận/từ chối (lý do + similarity + face size).

Điểm móc (hook) — LƯU Ý: đều nằm SAU bước upscale/detailer:
- `image_gen_service.step3_upscale_export` — sau khi upscale + detailer từng ảnh: task 1-nhân-vật → `maybe_collect(..., source="approved")` (vì user đã bấm Chấp nhận mới tới Trạm 3).
- `batch_video_runner` (Phase D) — sau post-process từng frame: cảnh 1-nhân-vật → `maybe_collect(..., source="auto")`. **KHÔNG có tương tác user trong batch** — cổng chất lượng là bộ lọc duy nhất, đó là lý do ngưỡng auto phải khắt khe (0.65).
- KHÔNG hook ở Trạm 2 / ảnh draft nữa (đã bỏ quyết định cũ).

### A5. Bootstrap ref tự động cho nhân vật CHƯA có ảnh trong CW (quyết định 07/2026)

Vấn đề: batch tự động gặp nhân vật mới (đã có trong CW nhờ bóc tách nhân vật, nhưng chưa có `ref.*`) → các cảnh của nhân vật đó không có identity, mỗi cảnh một mặt.

Giải pháp — "ảnh đạt chuẩn đầu tiên làm ref":
1. Khi sinh cảnh có primary_character chưa có ref: sinh bình thường (không identity), sau upscale đưa ảnh qua cổng chất lượng RÚT GỌN (mặt detect được ≥160px; KHÔNG check similarity vì chưa có ref để so).
2. Ảnh ĐẦU TIÊN đạt cổng → crop vùng mặt (bbox × 1.6, pad vuông) lưu làm `characters/<slug>/ref.png` + đánh dấu `Character.ref_source = "auto_bootstrap"` (field mới, default `"manual"`); ảnh full cũng vào dataset.
3. TỪ CẢNH KẾ TIẾP trở đi, nhân vật này dùng ref đó qua IP-Adapter như nhân vật bình thường → đồng nhất từ ảnh thứ 2.
4. Nếu hết batch mà không ảnh nào đạt cổng → nhân vật không có ref, ghi vào BatchReport (`characters_no_ref`).
5. UI Context Window: nhân vật có `ref_source == "auto_bootstrap"` hiển thị badge "🤖 ref tự động" + nút "Chọn lại ref" (user thay bằng ảnh khác trong dataset hoặc upload) — vì ảnh đầu tiên do seed quyết định, user có quyền phủ quyết sau batch.
6. Thứ tự cảnh ảnh hưởng ref: cảnh ĐẦU TIÊN của nhân vật nên là cảnh mặt rõ nhất có thể — nếu LLM (Phase C) đánh dấu `shot_type`, batch runner ưu tiên sinh cảnh close/medium ĐẦU TIÊN của nhân vật đó trước các cảnh wide (chỉ hoán đổi thứ tự SINH, không đổi thứ tự ghép video).

Rủi ro chấp nhận được (đã thống nhất với chủ dự án): identity của nhân vật bootstrap do ảnh đầu quyết định — nếu ảnh đầu không ưng, user "Chọn lại ref" rồi reroll các cảnh của nhân vật đó.

### A4. Nghiệm thu Phase A
- Tạo nhân vật mới với 5 ảnh upload → dataset có 5 `approved_*`, ref.* đúng ảnh đầu.
- Sinh 4 ảnh nhân vật đó → dataset tăng thêm ảnh `auto_*` (nếu similarity đạt), không vượt 15 khi chưa duyệt.
- Mở context.json cũ (backup trước) → load không crash.
- Bootstrap ref (A5): xoá ref của 1 nhân vật test → chạy sinh 3 cảnh có nhân vật đó → sau cảnh đạt chuẩn đầu tiên, `ref.png` xuất hiện với `ref_source="auto_bootstrap"`, cảnh sau có log áp IP-Adapter.
- `py_compile` sạch toàn bộ file sửa.

---

## PHASE B — LORA LIFECYCLE GẮN VỚI PHIÊN SINH HÀNG LOẠT

Mục tiêu: train đúng lúc, không chặn sáng tạo; nhân vật phụ bỏ qua được.

### B1. Train service (bọc script CLI hiện có)
File mới: `app/services/storytelling/lora_trainer.py`
```python
def build_instance_prompt(character) -> str   # instance_prompt nếu có, không thì clean từ keywords_en (bỏ tag góc chụp, lấy ≤12 tag ngoại hình)
def train_character(ctx_mgr, slug, steps=None, progress_cb=None) -> bool
def get_trainable_characters(ctx_mgr, slugs: list[str]) -> list[dict]
    # trả [{slug, name, n_images, eligible(bool: n_images>=8), lora_exists}]
```
`train_character` làm tuần tự:
1. `StorytellingPipeline().release()` — BẮT BUỘC giải phóng VRAM trước.
2. Set `lora_status="training"`, save context.
3. Gọi `subprocess` chạy `scripts/train_character_lora.py` bằng `sys.executable` (chính venv đang chạy), `--images_dir = dataset_dir`, `--instance_prompt = build_instance_prompt(...)`, `--steps`: 600 nếu <10 ảnh, 800 nếu 10–19, 1000 nếu ≥20. Stream stdout ra `progress_cb` (đọc line "step X/Y").
4. Thành công (file `resource/character_loras/<slug>.safetensors` tồn tại) → `lora_status="trained"`, `lora_trained_at=now`. Thất bại → `"failed"` + log stderr.
5. KHÔNG tự warmup lại pipeline (lần sinh kế tiếp tự warmup).

Lưu ý cho script train: hiện script nhận `--checkpoint` — truyền `ctx.checkpoint` của bộ truyện để LoRA khớp base model. Nếu user đổi checkpoint sau khi train → hiện cảnh báo trong UI (so sánh checkpoint lúc train, lưu thêm file `<slug>.json` cạnh LoRA ghi `{checkpoint, steps, n_images, trained_at}`).

### B2. Hộp thoại tiền-batch ("Pre-flight check")
Vị trí: `webui/Main.py`, ngay khi bấm "▶ Sinh Tất Cả Ảnh" (batch ảnh) và nút chạy batch video (Phase D).
Luồng:
1. Quét `primary_character` của mọi task trong batch → tập nhân vật xuất hiện.
2. Gọi `get_trainable_characters(...)`. Phân loại:
   - `trained` → dùng LoRA (tự động, không hỏi).
   - `eligible` (đủ ≥8 ảnh, chưa train) → hiện checkbox "Train trước khi sinh (~30-45 phút/nhân vật)" — mặc định KHÔNG tick.
   - không eligible → dòng thông tin "thiếu ảnh (x/8) — sẽ dùng IP-Adapter".
3. Hiển thị bằng `st.expander("🧬 Kiểm tra LoRA nhân vật")` + form; nút "Bắt đầu sinh" thực thi: train tuần tự các nhân vật được tick (progress bar riêng từng nhân vật) → xong mới chạy batch.
4. State lưu `st.session_state["preflight_done"]` để rerun không hỏi lại trong cùng phiên.
Quy tắc bất biến: batch KHÔNG BAO GIỜ bị chặn vì thiếu LoRA — mọi nhánh đều đi tiếp với IP-Adapter hoặc không identity.

### B3. Nghiệm thu Phase B
- Nhân vật đủ ảnh, tick train → LoRA xuất hiện trong `resource/character_loras/`, ảnh sinh sau đó có log "Đã load LoRA nhân vật".
- Không tick → batch chạy ngay bằng IP-Adapter, không lỗi.
- Train fail giữa chừng (giả lập: images_dir rỗng) → `lora_status="failed"`, batch vẫn chạy tiếp.
- 2 lần bấm sinh liên tiếp không hỏi lại pre-flight trong cùng session.

---

## PHASE C — TÁCH CẢNH NGỮ NGHĨA BẰNG LLM

Mục tiêu: cảnh 8–15s theo ngữ cảnh; 1 cảnh có thể trùm nhiều block SRT; giữ thuật toán cũ làm fallback.

### C1. Module mới: `app/services/storytelling/semantic_scene_splitter.py`
API:
```python
def split_scenes_semantic(md_text: str, total_audio_duration: float,
                          min_scene_sec=8, max_scene_sec=15) -> list[SemanticScene] | None
# None = LLM fail → caller fallback về md_parser cũ
```
`SemanticScene`: `{scene_index, text_vi (nguyên văn ghép các đoạn thuộc cảnh), summary_vi, location, characters: [tên], time_of_day, action}`.

Prompt LLM (system, tiếng Việt, output JSON):
- Input: toàn bộ kịch bản đã đánh số đoạn `[0], [1], [2]...` (đánh số theo paragraph sau khi lọc quảng cáo/H1 — tái dùng logic lọc của `md_parser`).
- Yêu cầu output: `{"scenes": [{"paragraphs": [0,1,2], "location": "...", "characters": ["..."], "action": "...", "summary": "..."}]}` — mỗi cảnh là dãy đoạn LIÊN TIẾP, không bỏ sót, không chồng lấn, ước lượng mỗi cảnh đọc ~8–15 giây (dựa tỷ lệ số từ / tổng từ × tổng thời lượng audio được cung cấp trong prompt).
- Ranh giới cảnh = đổi địa điểm, đổi nhóm nhân vật, đổi hành động chính, hoặc chuyển thời gian.
- Kịch bản dài: chunk theo 6000 từ, chunk sau nhận kèm 2 cảnh cuối của chunk trước làm ngữ cảnh nối; ghép kết quả, đánh lại scene_index.
Validation bắt buộc sau khi parse JSON (fail bất kỳ mục nào → return None để fallback):
- Mọi paragraph index xuất hiện đúng 1 lần, liên tiếp, phủ kín 0..N-1.
- Số cảnh ≥ 1 và ≤ số đoạn.

### C2. Post-rules ép nhịp cảnh (thuần thuật toán, sau LLM)
- Ước thời lượng cảnh = (số từ cảnh / tổng từ) × tổng thời lượng audio.
- Cảnh < `min_scene_sec` → gộp vào cảnh liền sau (hoặc trước nếu là cảnh cuối); cập nhật characters = hợp 2 tập.
- Cảnh > `max_scene_sec × 1.8` → cắt đôi tại ranh giới đoạn gần điểm giữa nhất (location/characters giữ nguyên, action thêm "(tiếp)").

### C3. Map SRT theo nội dung (thay map tuyến tính)
Sửa `srt_mapper.py`, thêm hàm `map_semantic_scenes_to_srt(scenes, srt_blocks)`:
1. Chuẩn hoá văn bản 2 phía: lowercase, bỏ dấu câu, NFKD bỏ dấu tiếng Việt.
2. Ghép toàn bộ SRT blocks thành 1 chuỗi từ, ghi lại (block_index, vị_trí_từ_bắt_đầu).
3. Với từng cảnh (theo thứ tự): dùng `difflib.SequenceMatcher` tìm đoạn khớp của `text_vi` (chuẩn hoá) trong chuỗi SRT **bắt đầu từ con trỏ hiện tại** (không quay lui — cảnh và audio cùng thứ tự tuyến tính). Lấy block chứa từ khớp đầu → `start_time`; block chứa từ khớp cuối → `end_time`; đẩy con trỏ tới sau block cuối.
4. Ratio khớp < 0.5 → gán cảnh theo tỷ lệ số từ (nội suy giữa end cảnh trước và start vùng khớp của cảnh sau tìm được gần nhất). Log warning từng cảnh gán nội suy.
5. Bảo đảm bất biến: start/end tăng dần, không chồng lấn, cảnh cuối end = duration audio.
Trường hợp không có SRT: giữ nguyên 2 nhánh hiện tại (Whisper sinh SRT, hoặc chia theo tỷ lệ từ).

### C4. Tích hợp vào orchestrator
`orchestrator.step1_generate_script`:
```python
scenes = split_scenes_semantic(md_text, dur)          # thử LLM
if scenes is None: scenes = parse_md_to_scenes(...)   # fallback cũ (giữ nguyên)
else: scenes = convert_to_Scene_dataclass(...)        # điền text_vi, characters_in_scene, primary_character theo match tên với context
scenes = map_semantic... / map_scenes_to_timeline(...)
```
- `primary_character` chọn theo: nhân vật trong `scene.characters` có identity (`ctx_mgr.has_identity`) và xuất hiện đầu tiên trong action.
- `llm_prompter.generate_prompts_batch` giữ nguyên nhưng nhận thêm `scene.location/action/summary` đưa vào user prompt ("Director's Note") để prompt ảnh bám cảnh.
- UI thêm toggle "🧠 Tách cảnh thông minh (LLM)" mặc định BẬT, tắt = dùng thuật toán cũ; hiển thị bảng preview cảnh (index, thời lượng ước tính, location, nhân vật, action) TRƯỚC khi sang bước sinh ảnh để user sửa tay (cho phép sửa action/location từng cảnh — text_input trong data_editor).

### C5. Nghiệm thu Phase C
- Kịch bản mẫu 3 chương (dùng `storage/contexts/Nguoi_Tren_Van_Nguoi` data thật): số cảnh giảm còn ~duration/11s ±30%; không cảnh nào <5s hoặc >30s.
- Tắt mạng LLM → tự fallback thuật toán cũ, không crash.
- SRT lệch (thiếu 2 block đầu) → cảnh vẫn gán timing tăng dần, có warning.

---

## PHASE D — PIPELINE SINH VIDEO HÀNG LOẠT END-TO-END

Mục tiêu: bỏ N cặp (kịch bản.md + audio.mp3/wav [+ .srt]) vào 1 thư mục → nhận N video, có resume, có báo cáo.

### D1. Quy ước input
Thư mục batch: mỗi item là 1 thư mục con HOẶC bộ file trùng tên (`ep01.md` + `ep01.mp3` [+ `ep01.srt`]). Scan bằng glob, ghép theo stem. Item thiếu audio → bỏ qua + ghi báo cáo.

### D2. Module mới: `app/services/storytelling/batch_video_runner.py`
```python
def run_batch(ctx_mgr, items: list[BatchItem], options, progress_cb) -> BatchReport
```
- Với từng item, chạy tuần tự: `step1` (Phase C) → pre-flight LoRA (Phase B — chỉ hỏi 1 LẦN cho cả batch, trước item đầu: quét nhân vật của TẤT CẢ kịch bản) → `step2` sinh ảnh → `step3` upscale (theo config) → `assemble_video` → lưu `storage/tasks/<uuid>/final.mp4` rồi copy sang thư mục output đặt tên `<stem>_final.mp4`.
- State: file `batch_state.json` trong thư mục output: `{items: {stem: {status: pending|script_done|images_done|video_done|failed, task_dir, error}}}`. Khởi động lại → bỏ qua item `video_done`, resume item dở theo `status` (tận dụng save_state/load_state sẵn có của orchestrator cho từng item).
- Error handling: mọi exception của 1 item → ghi `failed` + traceback vào report, `release()` pipeline nếu nghi OOM (bắt `torch.cuda.OutOfMemoryError` riêng: giảm batch nội bộ, retry 1 lần), đi tiếp item sau. KHÔNG raise xuyên batch.
- Nhân vật mới chưa có LoRA/ảnh ref gặp giữa batch: đúng quyết định đã chốt — sinh không identity + đánh dấu cảnh đó trong report (`scenes_no_identity: [...]`) để user reroll sau.
- `BatchReport` cuối: bảng per-item (thời gian, số cảnh, số cảnh thiếu identity, đường dẫn video, lỗi nếu có) — hiển thị UI + ghi `batch_report.json`.

### D3. UI
Tab 5, chế độ thực thi mới: "📦🎬 Sinh Video Hàng Loạt (Batch từ thư mục)":
- Input: đường dẫn thư mục (text_input), nút Scan → bảng item tìm thấy.
- Pre-flight LoRA (Phase B) hiển thị sau Scan.
- Nút "▶ Chạy Batch" → progress 2 tầng (item x/N + bước trong item), nút "⏹ Dừng sau item hiện tại" (set flag, runner kiểm tra giữa các item).
- Kết thúc: hiển thị BatchReport + link mở thư mục output.

### D4. Nghiệm thu Phase D
- Batch 2 item nhỏ (audio 1-2 phút): ra 2 video có phụ đề, đúng fps/codec config.
- Giết app giữa item 2 → chạy lại → item 1 bị bỏ qua (done), item 2 resume không sinh lại ảnh đã có.
- Item hỏng (md rỗng) → failed trong report, item sau vẫn chạy.

---

## PHASE E — KIỂM THỬ TỔNG & TÀI LIỆU

1. Smoke test tự động `apps/MediaComposer/tests/test_storytelling_smoke.py` (chạy không cần GPU):
   - Import mọi module storytelling; `_clamp_prompt_words`; validation của semantic splitter với JSON giả (đủ/thiếu/chồng lấn paragraph); `map_semantic_scenes_to_srt` với SRT giả (khớp hoàn hảo, lệch, thiếu); FIFO của `add_dataset_image` (ảnh PIL giả); load context.json phiên bản cũ.
   - Chạy bằng: `..\..\..venv\Scripts\python.exe -m pytest tests/ -x` (bổ sung pytest vào requirements nếu chưa có).
2. Test thủ công end-to-end trên GPU theo checklist các mục Nghiệm thu A→D.
3. Cập nhật `README.md` (mục Workflow 5) + `PROJECT_SUMMARY.md`: luồng batch video, LoRA lifecycle, cấu trúc dataset.
4. Nhắc user commit: `resource/character_loras/*.safetensors` + `*.json` metadata (dùng được ngay trên máy khác sau khi pull, vì loader chỉ cần file tồn tại).

---

## RỦI RO & QUYẾT ĐỊNH MỞ (Agent thực hiện cần lưu ý)

| Rủi ro | Đối sách |
|---|---|
| LLM local trả JSON sai định dạng thường xuyên | Validation chặt + fallback thuật toán cũ ở MỌI điểm gọi LLM; log mẫu response hỏng vào `storage/logs/llm_errors/` để tinh chỉnh prompt |
| CLIP similarity 0.60 nhận ảnh sai nhân vật (2 nhân vật cùng tông màu) | Ngưỡng nằm trong config (`dataset_collect_threshold`); ảnh auto có prefix riêng, xoá được 1 nút; LoRA hỏng thì xoá file + xoá auto images + train lại từ approved |
| Train LoRA trên máy 6GB OOM khi Streamlit còn giữ VRAM | `release()` trước train (đã quy định); nếu vẫn OOM → chạy train qua `train_lora.bat` ngoài app (hướng dẫn trong UI) |
| LoRA train trên checkpoint A dùng với checkpoint B | Metadata `<slug>.json` + warning UI khi mismatch (B1) |
| Đổi tên/slug nhân vật làm mồ côi dataset & LoRA | Khi rename: di chuyển thư mục dataset + đổi tên file LoRA, hoặc chặn rename khi đã có LoRA (chọn cách 2 cho đơn giản, hiện message) |
| SequenceMatcher chậm với audio >30 phút | Giới hạn cửa sổ tìm kiếm ±2000 từ quanh vị trí nội suy; đo thời gian trong smoke test |
| Streamlit rerun giữa lúc train | Train chạy trong thread riêng + `st.status`; flag trong session_state; nút Dừng chỉ có tác dụng giữa các nhân vật |

## THỨ TỰ THỰC HIỆN & ƯỚC LƯỢNG

| Phase | Nội dung | Ước lượng | Phụ thuộc |
|---|---|---|---|
| A | Dataset nhân vật + thu thập tự động | 0.5–1 ngày | — |
| B | LoRA lifecycle + pre-flight | 1 ngày | A |
| C | Tách cảnh ngữ nghĩa + map SRT | 1–1.5 ngày | — (song song A/B được) |
| D | Batch video runner + UI + resume | 1–1.5 ngày | B, C |
| E | Test + docs | 0.5 ngày | A–D |

Nguyên tắc chung cho Agent: mỗi Phase là 1 lần bàn giao chạy được — KHÔNG gộp nhiều Phase vào 1 đợt sửa; không refactor ngoài phạm vi; giữ nguyên hành vi các workflow 1–4 và 6; mọi tính năng mới đều có đường tắt/fallback về hành vi cũ.
