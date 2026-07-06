# KẾ HOẠCH: ToolAutoMakeCartoonVideo2DFromComics — App tổng phủ lên AIVoice + toolCaoTruyen

> Bản 2.0 (07/07/2026) — THAY THẾ hoàn toàn plan cũ "implementation_plan_Tool_Make_money.md".
> Viết cho Agent thực hiện. ĐỌC HẾT mục 0 và 1 trước khi viết dòng code đầu tiên.
> File này đặt tạm trong AIVoice — Phase R0 sẽ chuyển nó về repo app tổng.

---

## 0. QUYẾT ĐỊNH ĐÃ CHỐT VỚI CHỦ DỰ ÁN (không hỏi lại)

1. **Cấu trúc**: DI CHUYỂN cả 2 dự án vào trong `ToolAutoMakeCartoonVideo2DFromComics/` (xem R0 — có quy trình move an toàn, KHÔNG move tay bừa).
2. **Xóa engine AIVoice**: valtec, RVC, Local GGUF LLM (local_ai_spice) — xóa như plan cũ Phase 1, làm trên nhánh git riêng.
3. **Bản gốc tiếng Trung/Anh sau dịch**: nén thành `_original.zip` trong thư mục bộ truyện rồi xóa file gốc.
4. **Giao tiếp**: MỘT web UI tổng duy nhất. Các bước gọi **hàm/CLI headless bên trong dự án con qua venv riêng của nó** (subprocess) — KHÔNG khởi động Streamlit/Flask UI của dự án con. Ưu điểm quyết định: subprocess kết thúc = VRAM tự giải phóng 100%, không cần dọn thủ công.
5. **Gemini-API local (cookie proxy, port 7860) LUÔN được khởi động cùng app tổng** — vì cả dịch (toolCaoTruyen) lẫn LLM prompt (MediaComposer) đều gọi nó.
6. **Luồng người dùng 3 bước**: Chọn truyện (cào tự động HOẶC chọn thư mục có sẵn, kèm toggle "cần dịch?") → Chọn giọng đọc (engine + tinh chỉnh + preset chất lượng) → Tạo video (MediaComposer AIStorytelling 2D, chế độ batch).
7. **Storage tập trung tại app tổng** (chi tiết mục 2).

## 0.1. Bối cảnh 2 dự án con (Agent PHẢI tự đọc lại trước khi code — liệt kê để định hướng)

**AIVoice** (`F:\programfiles\AIVoice`):
- TTS engines: edge, piper, xtts clone, kokoro, vieneu (GIỮ) | valtec, rvc, local_ai_spice (XÓA).
- `apps/MediaComposer`: Workflow 5 AIStorytelling đã nâng cấp mạnh 07/2026: `batch_video_runner.run_batch()` (resume + report + pre-flight LoRA), semantic scene split bằng LLM, IP-Adapter CLIP identity, face detailer có cổng kiểm định, bootstrap ref tự động, SRT tự sinh từ kịch bản (không cần Whisper), shot-type. **ĐÂY LÀ LÕI BƯỚC VIDEO — KHÔNG VIẾT LẠI, CHỈ GỌI.**
- `setup.bat` đã tự nhận GPU (RTX50→cu128, NVIDIA khác→cu124), tự patch basicsr, tự tải model MC. **TÁI DÙNG cho setup tổng.**
- Cấu hình sinh ảnh: `apps/MediaComposer/config.toml` (một nguồn sự thật steps/guidance/identity...).

**toolCaoTruyen** (`F:\programfiles\toolCaoTruyen`):
- `core/crawler_engine.py` + `sources/` (shuba69, metruyenchuvn, registry) — cào truyện.
- `translator/` (gemini_api_translator = API key, gemini_translator = cookie qua Gemini-API server, glossary_manager, languages).
- `Gemini-API/` — server proxy chạy port 7860 (venv/cách chạy riêng — Agent đọc `Gemini-API/server`, `run.bat` gốc để biết lệnh khởi động chuẩn).
- `app.py` (web UI cũ ~41KB — sẽ KHÔNG dùng trong app tổng, giữ nguyên để dùng độc lập), `main.py` (CLI hiện có — nền tốt cho adapter).

---

## 1. KIẾN TRÚC APP TỔNG

### 1.1. Cấu trúc thư mục (sau R0)
```
ToolAutoMakeCartoonVideo2DFromComics/
├── AIVoice/                    # dự án con (venv riêng)
├── toolCaoTruyen/              # dự án con (venv riêng, chứa Gemini-API/)
├── orchestrator/               # code app tổng (venv TỔNG riêng, SIÊU NHẸ — fastapi+uvicorn+httpx, KHÔNG torch)
│   ├── main.py                 # FastAPI :3000 — serve UI + API + SSE progress
│   ├── process_manager.py      # start/stop/health subprocess (Gemini-API, adapters)
│   ├── pipeline.py             # máy trạng thái luồng: crawl→translate→tts→video, per-story
│   ├── storage.py              # quản lý storage/ (mục 2), slug hoá tên bộ truyện
│   └── config.py               # đọc/ghi configs/global_config.json
├── webui/                      # HTML/CSS/JS tĩnh (1 trang wizard + settings), SSE cập nhật tiến trình
├── configs/global_config.json  # đường dẫn 2 dự án con, ports, API keys, cookies, defaults
├── storage/                    # mục 2
├── setup.bat                   # gọi setup.bat của 2 dự án con + tạo venv tổng
└── run.bat                     # khởi động: venv tổng → main.py → mở browser :3000 → tự bật Gemini-API
```

### 1.2. Nguyên tắc process & VRAM (QUAN TRỌNG NHẤT)
- Orchestrator (venv tổng) KHÔNG import torch/heavy lib — chỉ điều phối.
- Mỗi bước nặng = 1 subprocess chạy bằng `<du_an_con>\.venv\Scripts\python.exe <adapter_cli> ...` với `cwd` = thư mục dự án con (giữ mọi đường dẫn tương đối cũ hoạt động).
- **Subprocess exit = VRAM về 0** — không cần API dọn VRAM phức tạp như plan cũ. Chỉ cần bảo đảm: chạy TUẦN TỰ, bước sau chỉ start khi bước trước exit.
- Gemini-API: process nền sống suốt phiên (không dùng VRAM — chỉ là proxy cookie). Health check `GET :7860` mỗi 30s, chết thì tự restart, `taskkill /T /PID` khi app tổng thoát (atexit + đóng cửa sổ).
- Progress protocol thống nhất cho MỌI adapter: in ra stdout từng dòng JSON `{"pct": 0-100, "msg": "...", "stage": "..."}`, kết thúc in `{"done": true, "result": {...}}` hoặc `{"error": "..."}`. Orchestrator parse → đẩy SSE cho UI. Exit code ≠ 0 = fail. **Ép `PYTHONIOENCODING=utf-8` khi spawn** (log tiếng Việt trên Windows console sẽ vỡ nếu quên — lỗi kinh điển).

### 1.3. Ports
| Service | Port | Ghi chú |
|---|---|---|
| Web UI tổng | 3000 | FastAPI/uvicorn |
| Gemini-API | 7860 | luôn bật cùng app |
| (Streamlit MC, Flask AIVoice) | 8502, 5000 | KHÔNG dùng trong app tổng; vẫn chạy độc lập được khi dev |

---

## 2. STORAGE TẬP TRUNG (đặc tả chốt)

```
storage/
├── tasks/                                  # working data MỌI bước (thay cho MC storage/tasks)
│   └── <task_uuid>/...                     # cấu trúc bên trong giữ nguyên như MC hiện tại
└── truyen/
    └── <Ten_Bo_Truyen_slug>/               # slug ASCII an toàn Windows (mục 2.1)
        ├── story.json                      # metadata: tên gốc đầy đủ, nguồn cào, ngôn ngữ, trạng thái từng bước, ngày
        ├── raw/                            # chương đã sẵn sàng dùng (TIẾNG VIỆT)
        │   ├── chuong_0001.md              # sau dịch (hoặc nguyên bản nếu đã là tiếng Việt)
        │   ├── chuong_0001.wav             # audio TTS — TRÙNG STEM với .md (bắt buộc, để scan_batch_dir ghép cặp)
        │   └── _original.zip               # bản gốc Trung/Anh nén lại trước khi xóa
        └── video/                          # output MediaComposer: chuong_0001.mp4 + batch_report.json
```

### 2.1. Quy tắc đặt tên (chống lỗi Windows/Unicode)
- Thư mục bộ truyện: slug hoá (bỏ dấu, thay ký tự cấm `<>:"/\|?*` bằng `_`, giới hạn 80 ký tự). Tên gốc đầy đủ lưu trong `story.json` để UI hiển thị đẹp.
- File chương: `chuong_%04d.md` đánh số theo thứ tự cào — KHÔNG dùng tên chương làm tên file (tên chương chứa ký tự cấm + dài).

### 2.2. Redirect MediaComposer (bắt buộc sửa, có kiểm soát)
MC hiện hardcode `_mc_root/storage/tasks` (orchestrator.py step1, batch_video_runner) và contexts tại `_mc_root/storage/contexts`.
- Thêm **biến môi trường `MC_STORAGE_TASKS`**: các điểm tạo `task_dir` trong MC ưu tiên `os.environ.get("MC_STORAGE_TASKS")` trước đường dẫn mặc định. App tổng set env này khi spawn adapter = `storage/tasks` của app tổng. Chạy MC độc lập (không env) → hành vi cũ nguyên vẹn. **KHÔNG đổi contexts** (Context Window, LoRA, dataset nhân vật ở lại trong MC — chúng là "tri thức" của MC, không phải dữ liệu phiên).
- Video output: `run_batch(output_dir=...)` ĐÃ là tham số — truyền `storage/truyen/<slug>/video`, không cần sửa gì.
- Điểm phải grep khi làm: `storage/tasks` xuất hiện trong `orchestrator.py`, `batch_video_runner.py` (MC) — sửa cả 2, py_compile sau mỗi file.

---

## 3. CÁC PHASE THỰC HIỆN (R0 → R8, mỗi phase bàn giao chạy được)

### R0 — Di chuyển an toàn + khung app tổng (0.5 ngày)
1. Commit sạch cả 2 repo con (bắt user xác nhận đã commit/push trước khi move).
2. Tạo `F:\programfiles\ToolAutoMakeCartoonVideo2DFromComics\`, MOVE 2 thư mục dự án vào (robocopy /MOVE hoặc move — cùng ổ F: nên là rename tức thời).
3. **BẪY VENV SAU KHI MOVE (lỗi chắc chắn gặp)**: venv Windows ghi đường dẫn tuyệt đối trong `pyvenv.cfg` và shebang của `Scripts\pip.exe` → sau move, `python.exe` của venv vẫn chạy nhưng **`pip.exe` sẽ hỏng**. Đối sách: KHÔNG gọi `pip.exe` trực tiếp nữa — mọi lệnh pip trong setup/docs đổi thành `python.exe -m pip`. Kiểm tra sau move: chạy `AIVoice\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"` và tương tự cho toolCaoTruyen. Nếu venv hỏng nặng → xóa venv, chạy lại setup.bat con (đã tự cài đủ).
4. Tạo khung thư mục orchestrator/webui/configs/storage + `run.bat`, `setup.bat` tổng (khung rỗng).
5. Nghiệm thu: 2 dự án con chạy độc lập BÌNH THƯỜNG từ vị trí mới (AIVoice run.bat, toolCaoTruyen run.bat); git 2 repo còn nguyên.

### R1 — Xóa engine thừa AIVoice (0.5–1 ngày, nhánh git `slim-engines`)
Theo plan cũ Phase 1, cập nhật theo hiện trạng:
- DELETE: `src/engines/rvc_engine.py`, `src/engines/valtec.py`, `src/utils/local_ai_spice.py`.
- MODIFY `requirements.txt`: bỏ `llama-cpp-python`, `fairseq`, `faiss-cpu`, `praat-parselmouth`, `pyworld`, `torchcrepe`, và dòng ghi chú rvc-python; bỏ mục cài `rvc-python --no-deps` trong `setup.bat`.
- MODIFY: `src/main.py`, `src/web_ui.py`, `src/engines/__init__.py`, `src/download_models.py` — gỡ mọi import/route/menu/download của 3 engine (grep `rvc|valtec|spice|llama` toàn src trước, xử lý hết từng kết quả).
- GIỮ: edge, piper, clone(XTTS), kokoro, vieneu; giữ `av`, `ffmpeg-python`... nếu engine giữ lại còn dùng (grep trước khi xóa lib nào khỏi requirements).
- Xóa thư mục model: `models/rvc/`, `models/llm/`, valtec checkpoints; `third_party/valtec-tts`.
- Nghiệm thu: `chay_kiem_thu.bat` (bộ test TTS có sẵn) pass với các engine còn lại; web_ui AIVoice mở được, không route chết; đo dung lượng giảm (kỳ vọng >10GB).

### R2 — Storage tập trung + redirect MC (0.5 ngày)
- Implement `orchestrator/storage.py` (slug, tạo cây thư mục, story.json CRUD).
- Sửa MC theo mục 2.2 (`MC_STORAGE_TASKS`).
- Nghiệm thu: chạy MC độc lập → tasks vẫn vào chỗ cũ; chạy với env → tasks vào storage tổng; run_batch xuất video vào thư mục chỉ định.

### R3 — Adapter cào + dịch (toolCaoTruyen) (1–1.5 ngày)
File mới `toolCaoTruyen/adapter_cli.py` (Agent đọc `main.py`, `core/crawler_engine.py`, `translator/*` trước — tái dùng hàm có sẵn, KHÔNG viết lại logic cào/dịch):
- `crawl`: `--source shuba69|metruyenchuvn --story-id/--url --from N --to M --out <storage/truyen/<slug>/raw>` → xuất `chuong_%04d.md` + cập nhật story.json (tên truyện, ngôn ngữ phát hiện). Progress JSON per chương.
- `translate`: `--dir <raw> --engine gemini_api|gemini_cookie --glossary ...` → dịch từng .md (tái dùng glossary_manager), xong: nén toàn bộ bản gốc vào `_original.zip` (zipfile, kiểm tra zip mở được + đủ số file TRƯỚC khi xóa gốc), ghi đè .md bằng bản dịch cùng tên. Progress per chương. Idempotent: chương đã dịch (đánh dấu trong story.json) thì bỏ qua → resume được.
- Lưu ý: dịch qua `gemini_cookie` yêu cầu Gemini-API :7860 sống — adapter check health đầu tiên, lỗi thì báo rõ "Gemini-API chưa chạy".
- Nghiệm thu: cào 3 chương truyện thật → dịch → raw/ chỉ còn .md tiếng Việt + _original.zip; chạy lại lệnh dịch → skip toàn bộ (resume).

### R4 — Adapter TTS (AIVoice) (1 ngày)
File mới `AIVoice/adapter_tts_cli.py` (tái dùng engines + utils/audio có sẵn):
- Input: `--dir <raw> --engine edge|piper|clone|kokoro|vieneu --voice ... --speed ... --volume ...` + tham số nâng cao (mục 5.2). Với mỗi `chuong_XXXX.md` chưa có `.wav` cùng stem → sinh `chuong_XXXX.wav` cạnh nó. Resume tự nhiên (skip file đã có).
- Preset chất lượng: `--preset stable|fast|quality` map sẵn bộ tham số từng engine (định nghĩa trong file JSON `AIVoice/configs/tts_presets.json` để chỉnh không cần sửa code).
- Chuẩn hoá đầu ra: LUFS normalize (pyloudnorm đã có), sample rate thống nhất 24k/44k1, xuất wav.
- Nghiệm thu: 3 md → 3 wav nghe được, chạy lại → skip; engine clone (XTTS) chạy xong process exit → nvidia-smi VRAM về ~0.

### R5 — Adapter Video (MediaComposer) (0.5 ngày — NHẸ vì run_batch có sẵn)
File mới `AIVoice/apps/MediaComposer/adapter_video_cli.py`:
- Input: `--dir <raw> --out <video> --story <slug> [--context <mc_context_slug>] [--upscale 0/1] [--burn-sub 0/1] [--fresh 0/1]`.
- Việc của adapter: đảm bảo context MC tồn tại cho bộ truyện (chưa có → tạo `ContextManager.create_context(tên truyện)` — nhân vật sẽ do bootstrap ref T7 tự lo trong batch); gọi `scan_batch_dir(raw)` → `run_batch(items, output_dir=video, options)`; forward progress_cb ra stdout JSON; in report cuối.
- Nghiệm thu: 2 cặp md+wav → 2 mp4 trong storage/truyen/<slug>/video + batch_report.json; giết giữa chừng → chạy lại resume.

### R6 — Orchestrator + config (1–1.5 ngày)
- `process_manager.py`: spawn/kill (taskkill /T), health Gemini-API + auto-restart, atexit cleanup, chỉ cho 1 job nặng chạy tại 1 thời điểm (lock).
- `pipeline.py`: máy trạng thái per-story đọc/ghi `story.json`: `crawled → translated → tts_done → video_done` (+ `failed_<step>` kèm lỗi). API: start step, get status, danh sách truyện. Job chạy nền (thread) + log ring-buffer cho SSE.
- `config.py` + `global_config.json`: đường dẫn 2 dự án con, ports, Gemini API keys, đường dẫn cookies.json (của Gemini-API), engine dịch mặc định, TTS preset mặc định, tham số video mặc định. API đọc/ghi + nút "Test kết nối" từng dịch vụ.
- Khởi động app: tự start Gemini-API (theo quyết định #5).
- Nghiệm thu: gọi các API bằng curl chạy trọn luồng 1 truyện 3 chương không cần UI.

### R7 — Web UI tổng (1–1.5 ngày)
Một trang wizard (HTML/JS thuần + SSE, dark mode; KHÔNG framework nặng):
- **Trang chính**: danh sách bộ truyện trong storage (từ story.json) + trạng thái từng bước + nút tiếp tục từ bước dở.
- **Bước 1 — Chọn truyện**: 2 tab (Cào mới: nguồn/ID/chương | Thư mục có sẵn: import vào storage). Toggle "Truyện cần dịch?" (tự phát hiện ngôn ngữ từ nội dung chương đầu, user override được).
- **Bước 2 — Giọng đọc**: chọn engine, voice, preset chất lượng, expander tinh chỉnh nâng cao (speed/volume/LUFS/sample-rate/khoảng lặng giữa câu), nút "🔊 Nghe thử" (sinh 1 câu mẫu qua adapter với `--sample`).
- **Bước 3 — Video**: các option map thẳng run_batch (upscale, phụ đề, làm mới) + hiển thị pre-flight LoRA dạng thông tin.
- **Settings**: toàn bộ global_config + trạng thái Gemini-API (sống/chết, nút restart).
- Progress: thanh 2 tầng (bước / chi tiết), log cuộn, đọc từ SSE.
- Nghiệm thu: chạy trọn luồng 1 truyện chỉ bằng chuột từ UI; đóng trình duyệt mở lại → trạng thái còn nguyên.

### R8 — setup.bat/run.bat tổng + kiểm thử máy RTX 5060 (0.5–1 ngày)
- `setup.bat` tổng: check Python 3.11/Git → tạo venv tổng (fastapi uvicorn httpx sse-starlette) → gọi `toolCaoTruyen\setup.bat` → gọi `AIVoice\setup.bat` (ĐÃ tự xử lý GPU cu124/cu128, basicsr, model MC — không làm lại) → ghi global_config.json mặc định. Mọi pip đều `python -m pip`.
- `run.bat`: venv tổng → `python orchestrator/main.py` → start Gemini-API → mở browser :3000.
- Kiểm thử trên máy RTX 5060: pull → setup.bat → run.bat → trọn luồng 3 chương. Ghi lại thời gian từng bước + VRAM đỉnh từng bước vào README.

---

## 4. LƯU Ý TRÁNH LỖI CHO AGENT (đúc kết từ các vòng sửa trước)

1. **Đọc trước khi sửa**: mở và đọc file thật, không tin mô tả trong plan cũ — 2 dự án đã thay đổi nhiều (đặc biệt MC 07/2026).
2. **py_compile sau MỖI file sửa**; mỗi phase chạy nghiệm thu của phase đó rồi mới sang phase sau.
3. **Không phá chế độ độc lập**: AIVoice web_ui, MC Streamlit, toolCaoTruyen app.py phải vẫn chạy được một mình sau mọi thay đổi (regression test cuối mỗi phase).
4. **Mọi subprocess**: `cwd` = thư mục dự án con; env kèm `PYTHONIOENCODING=utf-8`; đường dẫn truyền vào là TUYỆT ĐỐI (đã qua os.path.abspath ở orchestrator).
5. **Kill process tree** bằng `taskkill /F /T /PID` (uvicorn/chrome con của Gemini-API sẽ mồ côi nếu chỉ kill cha).
6. **Unicode**: tên bộ truyện tiếng Việt/Trung chỉ nằm trong story.json; mọi path trên đĩa là slug ASCII.
7. **_original.zip**: verify zip (mở lại, đếm file, thử đọc 1 file) TRƯỚC khi xóa bản gốc — xóa nhầm là mất dữ liệu phải cào lại.
8. **Không import chéo venv**: orchestrator tuyệt đối không import module của dự án con (khác venv, sẽ vỡ) — chỉ giao tiếp qua CLI/JSON.
9. **MC contexts/LoRA ở lại trong MC** — chỉ tasks + video output đi ra storage tổng.
10. **Windows long-path**: giữ tên task_uuid ngắn, tổng path < 240 ký tự (slug 80 + cấu trúc cố định là đủ an toàn).
11. **Nghiệm thu VRAM giữa các bước bằng `nvidia-smi`** — kỳ vọng về ~0 sau mỗi adapter exit (đây là lợi ích chính của kiến trúc subprocess, phải kiểm chứng).

## 5. ĐỀ XUẤT CẢI TIẾN (đã đưa vào plan, chủ dự án duyệt rồi mới làm thêm ngoài danh sách)

1. **Chế độ "Auto toàn tập"**: 1 nút chạy tuần tự cả 4 bước cho 1 bộ truyện (hoặc queue nhiều bộ chạy qua đêm) — pipeline.py đã là máy trạng thái nên chỉ là vòng lặp.
2. **Nghe thử giọng trước khi chạy cả bộ** (đã ghi ở R7 bước 2).
3. **Glossary per-bộ-truyện**: toolCaoTruyen có glossary_manager — lưu glossary vào thư mục bộ truyện để tên nhân vật dịch nhất quán giữa các chương (ảnh hưởng trực tiếp chất lượng match nhân vật ở MC).
4. Thông báo Windows (toast/tiếng beep) khi hoàn thành job dài.

## 6. THỨ TỰ & ƯỚC LƯỢNG

| Phase | Nội dung | Ước lượng |
|---|---|---|
| R0 | Move an toàn + khung | 0.5 ngày |
| R1 | Slim AIVoice | 0.5–1 ngày |
| R2 | Storage + redirect MC | 0.5 ngày |
| R3 | Adapter cào + dịch | 1–1.5 ngày |
| R4 | Adapter TTS | 1 ngày |
| R5 | Adapter video | 0.5 ngày |
| R6 | Orchestrator + config | 1–1.5 ngày |
| R7 | Web UI tổng | 1–1.5 ngày |
| R8 | Setup tổng + test RTX 5060 | 0.5–1 ngày |

Tổng: ~7–9 ngày làm việc. R1 và R2 độc lập nhau (làm song song được); R3–R5 độc lập nhau sau khi R2 xong; R6 cần R3–R5; R7 cần R6.
