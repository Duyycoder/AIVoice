# 🎙️ AIVoice: Khung Chuyển Đổi Văn Bản Thành Giọng Nói Tiếng Việt Đa Động Cơ

**AIVoice** là một framework chuyển đổi văn bản thành giọng nói (Text-to-Speech - TTS) tiếng Việt mạnh mẽ, mô-đun hóa, thiết kế theo các nguyên lý **Clean Architecture** hỗ trợ cả hai hình thức chạy ngoại tuyến (offline) và trực tuyến (online). Hệ thống hỗ trợ làm sạch tài liệu Markdown, tự động chia đoạn văn tránh tràn bộ nhớ GPU, tích hợp hậu xử lý âm thanh chuyên nghiệp (chuẩn hóa âm lượng LUFS, hiệu ứng Fade-in/Fade-out) và sao chép giọng nói (Voice Cloning) chỉ với một tệp âm thanh mẫu.

---

## 📌 Mục Lục
1. [📂 Sơ Đồ Cấu Trúc Dự Án (Clean Architecture)](#-sơ-đồ-cấu-trúc-dự-án-clean-architecture)
2. [🛠️ Hướng Dẫn Cài Đặt](#️-hướng-dẫn-cài-đặt)
3. [🚀 Hướng Dẫn Sử Dụng Chi Tiết](#-hướng-dẫn-sử-dụng-chi-tiết)
   * [Chế Độ Tương Tác (Wizard Mode)](#1-chế-độ-tương-tác-wizard-mode-khuyên-dùng)
   * [Chế Độ Dòng Lệnh CLI (Command Line Mode)](#2-chế-độ-dòng-lệnh-cli-command-line-mode)
   * [Các Ví Dụ Lệnh Thực Tế](#các-ví-dụ-lệnh-thực-tế)
4. [⚙️ Chi Tiết Hậu Xử Lý Âm Thanh & Phiên Âm](#️-chi-tiết-hậu-xử-lý-âm-thanh--phiên-âm)
5. [🖥️ Chẩn Đoán GPU & Cấu Hình NVIDIA CUDA](#️-chẩn-đoán-gpu--cấu-hình-nvidia-cuda)
6. [🔍 Lỗi Thường Gặp & Cách Khắc Phục (Troubleshooting)](#-lỗi-thường-gặp--cách-khắc-phục-troubleshooting)
7. [🔌 Hướng Dẫn Mở Rộng Động Cơ Mới (Developer Guide)](#-hướng-dẫn-mở-rộng-động-cơ-mới-developer-guide)
8. [🧪 Hệ Thống Kiểm Thử Tự Động](#-hệ-thống-kiểm-thử-tự-động)

---

## 📂 Sơ Đồ Cấu Trúc Dự Án (Clean Architecture)

```text
AIVoice/
├── configs/               # [1. CẤU HÌNH] Chỉ chứa thiết lập hệ thống
│   └── default.json       # Tệp cấu hình mặc định (Đổi tên từ config.json)
│
├── data/                  # [2. DỮ LIỆU THỰC TẾ] Nơi chứa tệp đầu vào, đầu ra của người dùng
│   ├── inputs/            # Thả truyện, văn bản .md, .txt vào đây để đọc
│   ├── outputs/           # File audio kết quả (.wav) được sinh ra tự động
│   └── voices/            # Chứa các file giọng nói mẫu (e.g., ref_voice.wav) để sao chép giọng
│
├── models/                # [3. TRỌNG SỐ MÔ HÌNH AI] (Không đưa lên Git)
│   ├── piper/             # Chứa model ONNX và file cấu hình JSON của Piper
│   ├── xtts_v2/           # Chứa mô hình XTTSv2 fine-tuned và các file phụ trợ
│   ├── llm/               # Chứa các mô hình GGUF LLM để viết lại cảm xúc (e.g., Qwen2.5)
│   └── rvc/               # Chứa mô hình RVC .pth và file .index dùng để chuyển đổi giọng nói

│
├── src/                   # [4. LÕI LOGIC] Nơi chứa toàn bộ mã nguồn xử lý logic
│   ├── engines/           # Các Động Cơ TTS bọc thành lớp độc lập
│   │   ├── base.py        # Giao diện abstract lớp cha
│   │   ├── edge.py        # Bộ điều khiển Microsoft Edge Online TTS
│   │   ├── piper.py       # Bộ điều khiển Piper ONNX TTS Offline
│   │   ├── clone.py       # Bộ điều khiển Coqui XTTSv2 Voice Cloning Offline
│   │   ├── rvc_engine.py  # Bộ chuyển đổi giọng nói Voice-to-Voice sử dụng RVC
│   │   ├── kokoro.py      # Bộ điều khiển Kokoro-Vietnamese Offline TTS
│   │   ├── valtec.py      # Bộ điều khiển Valtec-TTS Offline TTS
│   │   └── vieneu.py      # Bộ điều khiển VieNeu-TTS Offline/Online TTS
│   └── utils/             # Các thư viện bổ trợ
│       ├── audio.py       # Hậu xử lý âm lượng (LUFS), cắt ghép, fade-in/out
│       ├── text.py        # Làm sạch cú pháp markdown và phân tách câu (chunking)
│       ├── cache.py       # Bộ quản lý lưu trữ đệm Semantic Cache cho phân đoạn âm thanh
│       ├── phoneme.py     # Hỗ trợ chuyển đổi phiên âm sang IPA tiếng Việt
│       └── local_ai_spice.py # Viết lại văn bản tiếng Việt thêm cảm xúc/hài hước qua GGUF LLM cục bộ
│
├── tests/                 # [5. BỘ KIỂM THỬ] Độc lập để xác thực hệ thống
│   ├── test_data/         # Dữ liệu phục vụ kiểm thử (inputs/outputs cô lập)
│   └── run_tests.py       # Script chạy toàn bộ kịch bản kiểm thử tự động
│
├── check_gpu.py           # Công cụ kiểm tra sức khỏe và chẩn đoán GPU/CUDA
├── download_models.py     # Script tải xuống mô hình tự động (Piper & XTTSv2)
├── main.py                # [ENTRY POINT] CLI điều khiển chính (Hỗ trợ Wizard tương tác)
├── web_ui.py              # Giao diện Web UI (Flask-based SPA phong cách Glassmorphism)
├── requirements.txt       # Danh sách thư viện môi trường cần cài đặt
├── plan.md                # Kế hoạch phát triển dự án và trạng thái hiện tại
├── HUONG_DAN_SU_DUNG.md   # Hướng dẫn sử dụng tiếng Việt chi tiết cho Web UI & sửa lỗi RVC/phomie
├── chay_giao_dien.bat     # Tập lệnh khởi chạy nhanh máy chủ Web UI cục bộ (Windows)
├── chay_giao_dien.sh      # Tập lệnh khởi chạy nhanh máy chủ Web UI cục bộ (macOS/Linux)
├── chay_kiem_thu.bat      # Tập lệnh chạy nhanh bộ kiểm thử tích hợp (Windows)
├── chay_kiem_thu.sh       # Tập lệnh chạy nhanh bộ kiểm thử tích hợp (macOS/Linux)
├── kiem_tra_gpu.bat       # Tập lệnh chạy nhanh chẩn đoán GPU/CUDA (Windows)
├── kiem_tra_gpu.sh        # Tập lệnh chạy nhanh chẩn đoán GPU/CUDA (macOS/Linux)
├── setup.sh               # Tập lệnh cài đặt môi trường tự động (macOS/Linux)
├── MediaComposer/         # Công cụ đồng hành tạo video có phụ đề tự động (Streamlit)
└── README.md              # Tài liệu hướng dẫn sử dụng này
```

---

## 🛠️ Hướng Dẫn Cài Đặt

### 1. Yêu Cầu Hệ Thống
* **Hệ điều hành:** Windows 10/11.
* **Bộ xử lý đồ họa (GPU):** Khuyên dùng GPU NVIDIA (VRAM >= 6GB như RTX 3060, RTX 4060, RTX 5060) để tăng tốc nhân bản giọng nói XTTSv2.
* **Tối ưu phần cứng:** Dự án đã được tối ưu hóa cho card đồ họa thế hệ mới **RTX 5060 (8GB VRAM)** và tương thích ngược (RTX 30/40 series) thông qua chế độ FP16 Mixed Precision (giảm VRAM sử dụng xuống ~2GB) và tăng tốc TensorFloat-32 (TF32). Có cơ chế tự động chuyển đổi sang CPU nếu không có GPU rời.
* **Python:** Phiên bản 3.10 hoặc 3.11.

### 2. Quy Trình Cài Đặt Môi Trường Chi Tiết (4 Bước)

Để hệ thống chạy ổn định trên máy mới, bạn cần thực hiện theo các bước chuẩn sau (lưu ý: Bước cài đặt Build Tools chỉ cần làm duy nhất 1 lần trên mỗi máy, không cần lặp lại khi clone lại dự án):

#### **Bước 1: Cài đặt Python 3.11 (Tự động hoặc Thủ công)**
* **Tự động:** File **[setup.bat](setup.bat)** sẽ tự động phát hiện và cài đặt ngầm Python 3.11.9 cho bạn nếu máy chưa có. Bạn không cần làm gì ở bước này.
* **Thủ công (Nếu muốn):** Tải và cài đặt Python 3.11 từ trang chủ Python. Bắt buộc phải tích chọn ô **"Add Python to PATH"** khi chạy trình cài đặt.

#### **Bước 2: Cài đặt Microsoft C++ Build Tools (Bắt buộc cho Clone & RVC)**
* Vì động cơ Clone (XTTSv2) và RVC yêu cầu biên dịch mã nguồn C++ khi cài đặt, bạn cần cài đặt bộ biên dịch chính thức của Microsoft.
* **Đường dẫn tải:** Truy cập [Microsoft Downloads](https://visualstudio.microsoft.com/downloads/) → cuộn xuống mục **"All Downloads"** → chọn **"Tools for Visual Studio"** → Tải xuống **"Build Tools for Visual Studio"**.
* **Cách chọn Workload:** Khởi chạy installer, tích chọn ô **"Desktop development with C++"** (Phát triển ứng dụng Desktop với C++) rồi nhấn Install. Sau khi hoàn tất, hãy khởi động lại máy tính hoặc mở một cửa sổ Terminal mới.

#### **Bước 3: Cài đặt Git for Windows**
* Cài đặt Git nếu hệ thống chưa có để hỗ trợ quá trình tải trực tiếp các gói thư viện ngữ âm tiếng Việt từ GitHub.
* **Đường dẫn tải:** [Git for Windows](https://git-scm.com/download/win).
#### **Bước 4: Chạy script setup để thiết lập tự động**
* Kích hoạt terminal tại thư mục dự án trên máy mới.
* **Trên Windows:** Bấm đúp vào file **[setup.bat](setup.bat)** (hoặc chạy lệnh `setup.bat` từ Command Prompt/PowerShell).
* **Trên macOS/Linux:** Cấp quyền thực thi và chạy script `setup.sh`:
  ```bash
  chmod +x setup.sh && ./setup.sh
  ```
* Script sẽ tự động:
  1. Khởi tạo môi trường ảo `.venv`.
  2. Thiết lập đúng phiên bản pip tương thích (`pip 24.0`).
  3. Cài đặt toàn bộ dependencies trong `requirements.txt`.
  4. Cài đặt các gói ngữ âm tiếng Việt từ GitHub (nếu có Git).
  5. Tự động sao chép cấu hình `config.toml` từ file ví dụ trong `MediaComposer`.
  6. Tải tự động các mô hình AI Piper & XTTSv2.
  7. Chạy chẩn đoán GPU phần cứng CUDA.

*(Gợi ý: Nếu bạn muốn chạy thủ công từng lệnh thay vì dùng script tự động, bạn cũng có thể mở terminal và chạy tuần tự các lệnh trong file [setup.bat](setup.bat) hoặc [setup.sh](setup.sh) một cách thủ công).*

> [!TIP]
> Nếu bạn muốn sử dụng tính năng phiên âm ngữ âm IPA tiếng Việt (`--phonemize`), hãy cài đặt thêm các thư viện ngữ âm tùy chọn sau vào môi trường ảo:
> ```powershell
> .\.venv\Scripts\pip install git+https://github.com/vunb/viphoneme.git git+https://github.com/vunb/vinorm.git
> ```

### 3. Tải Xuống Các Mô HÌnh Lớn (Models Download)
Khung làm việc hoạt động theo tiêu chí **Offline First** (chạy ngoại tuyến trước). Để tải xuống các mô hình cần thiết vào đúng cấu trúc thư mục `models/` nhằm tránh tự động tải từ Internet trong lúc sinh âm thanh:

```powershell
# Tải mô hình Piper ONNX (mặc định vi_VN-vais1000-medium) vào models/piper/
python download_models.py --engine piper

# Tải mô hình XTTSv2 (kích thước ~2GB) vào models/xtts_v2/
python download_models.py --engine clone

# Tải toàn bộ mô hình (cả Piper và XTTSv2)
python download_models.py --engine all
```

---

## 🚀 Hướng Dẫn Sử Dụng Chi Tiết

Hệ thống cung cấp ba phương thức điều khiển: **Giao diện Web đồ họa (Web UI)**, **Chế độ tương tác từng bước (Wizard)** và **Chế độ dòng lệnh CLI**.

### 0. Giao Diện Web Đồ Họa (Web UI - Khuyên Dùng Cho Người Mới)

Để khởi chạy giao diện Web UI cục bộ vô cùng trực quan và đẹp mắt (Glassmorphism Dark Mode), hãy kích hoạt file script tương ứng:
* Chạy file `chay_giao_dien.bat` có sẵn trong thư mục dự án (bạn có thể copy ra màn hình Desktop hoặc bất kỳ vị trí nào để chạy mà không cần mở thư mục).
* Hoặc chạy lệnh sau trong terminal:
  ```powershell
  python web_ui.py
  ```
Sau khi chạy, giao diện sẽ tự động mở trên trình duyệt tại địa chỉ: [http://127.0.0.1:5000](http://127.0.0.1:5000).

#### Các Tính Năng Vượt Trội Trên Web UI:
1. **Lựa chọn Phần cứng (2 Option rõ ràng):** 
   - **GPU (NVIDIA CUDA) - Tối ưu:** Tự động kích hoạt FP16 Mixed Precision cho XTTSv2 giúp giảm dung lượng VRAM tiêu thụ từ ~3.5GB xuống còn ~2.0GB (hoàn toàn chạy được trên các card 6GB như RTX 3060) và bật chế độ tăng tốc TF32 trên các nhân Tensor Cores.
   - **CPU Mode - Tiêu chuẩn:** Tự động fallback về chạy hoàn toàn trên CPU ở độ chính xác FP32 tiêu chuẩn khi máy không có GPU rời.
2. **Hỗ trợ Thư mục/Tệp ngoài:** Cho phép bạn nhập trực tiếp đường dẫn tuyệt đối đến tệp tin văn bản cần xử lý hoặc thư mục lưu trữ ngoài dự án (ví dụ: `D:\truyen\chuong_1.md` hoặc `E:\audio_out`) mà không cần copy tệp vào thư mục dự án. Có tích hợp cơ chế kiểm tra quyền ghi và chặn các thư mục hệ thống nhạy cảm vì lý do bảo mật.
3. **Serialization Lock (Chặn OOM):** Web UI tích hợp cơ chế xếp hàng xử lý (`threading.Lock()`). Nếu bạn vô tình click đúp hoặc gửi nhiều yêu cầu cùng lúc từ các tab khác nhau, các yêu cầu sau sẽ hiển thị trạng thái `[Xếp hàng chờ xử lý (GPU đang bận)]` và chạy tuần tự, ngăn ngừa việc GPU bị quá tải dẫn đến crash/OOM.
4. **Console Log & Player tích hợp:** Hiển thị trực tiếp log xử lý thời gian thực, có danh sách tệp lịch sử và trình phát audio tích hợp để bạn nghe thử trực tiếp kết quả.
5. **Chẩn đoán GPU nhanh:** Có nút bấm chạy chẩn đoán nhanh `check_gpu.py` hiển thị báo cáo cấu hình CUDA/cuDNN ngay trên giao diện web.
6. **Tạo Video Tự Động (Tích hợp Media Composer):** Nút bấm khởi chạy công cụ tạo video đồng hành, tự động cấu hình ứng dụng Streamlit chạy nền tại cổng 8502 và mở trình duyệt để tạo video kèm phụ đề từ âm thanh thu được.

### 0.1. Tích hợp Tạo Video (Media Composer)
AIVoice hỗ trợ tích hợp với công cụ tạo video có phụ đề chạy giao diện Streamlit:
* **Khởi chạy từ Web UI:** Nhấp vào nút **"🎬 Tạo Video (Media Composer)"** ở thanh công cụ. Hệ thống sẽ tự động kích hoạt máy chủ Streamlit nền (tải các thư viện `moviepy`, `faster-whisper`, `openai`, `streamlit` trong môi trường ảo) và trỏ trình duyệt đến `http://127.0.0.1:8502`.
* **Cài đặt dependencies bổ sung:** Nếu bạn chưa có các gói thư viện này, vui lòng chạy lại script `setup.bat` để hệ thống tự động cập nhật và cài đặt đầy đủ các thư viện MediaComposer ở cuối tệp `requirements.txt`.


### 1. Chế Độ Tương Tác (Wizard Mode - KHUYÊN DÙNG)
Nếu bạn không muốn nhớ các tham số phức tạp, chỉ cần gõ lệnh sau:
```powershell
python main.py
```
Hệ thống sẽ hiển thị một menu tương tác bằng tiếng Việt trên terminal để bạn chọn từng bước:
1. **Chọn Engine / Chế độ chạy:** `edge` (Microsoft), `piper` (Offline nhẹ), `clone` (XTTSv2 sao chép giọng), `rvc` (Đổi giọng trực tiếp cho tệp âm thanh), hoặc `batch` (Chạy hàng loạt).
2. **Chọn Model:** Liệt kê các model sẵn có trong `models/piper/` hoặc `models/xtts_v2/`.
3. **Chọn Tệp đầu vào (Input):** Quét các file `.md`/`.txt` sẵn có trong `data/inputs/` để bạn chọn nhanh theo số thứ tự.
4. **Chọn Giọng nói / Ngôn ngữ:** Chọn giọng Nam/Nữ của Edge-TTS hoặc ngôn ngữ `vi`/`en` cho XTTSv2.
5. **Chọn File Âm Thanh Mẫu (Reference Audio):** Quét thư mục `data/voices/` để chọn file giọng mẫu clone.
6. **Nhập Tên File Đầu Ra:** Gợi ý mặc định theo tên file văn bản đầu vào.
7. **Cấu Hình Nâng Cao:** Cho phép thay đổi tốc độ đọc, bật/tắt phiên âm IPA, bật/tắt chuẩn hóa âm lượng LUFS, chỉnh độ dài fade-in/out, và khoảng lặng giữa các câu.

Sau khi hoàn tất, Wizard sẽ hiển thị lệnh tương đương và tự động chạy pipeline.

---

### 2. Chế Độ Dòng Lệnh CLI (Command Line Mode)

Dành cho việc tích hợp vào các hệ thống khác hoặc chạy tự động thông qua file script.

#### Danh Sách Tham Số CLI Chi Tiết
| Tham số | Phím tắt | Ý nghĩa | Giá trị mặc định |
| :--- | :--- | :--- | :--- |
| `--input` | Không | Đường dẫn đến tệp văn bản đầu vào `.md`/`.txt` (hoặc tệp âm thanh `.wav`/`.mp3`... khi sử dụng động cơ `rvc`). | *Bắt buộc (hoặc dùng `--input_dir`)* |
| `--input_dir` | Không | Đường dẫn đến thư mục chứa nhiều tệp `.md`/`.txt` (hoặc thư mục chứa tệp âm thanh khi dùng động cơ `rvc`). | *Bắt buộc (nếu không dùng `--input`)* |
| `--engine` | Không | Chọn động cơ sinh/chuyển đổi giọng nói (`edge`, `piper`, `clone`, `rvc`, `valtec`, `kokoro`, `vieneu`). | Lấy từ `configs/default.json` |
| `--model` | Không | Đường dẫn tới tệp/thư mục chứa trọng số mô hình. | Lấy từ `configs/default.json` |
| `--speed` | Không | Tốc độ đọc của giọng nói (ví dụ: `1.0`, `1.2`, `0.85`). | `1.0` |
| `--voice` | Không | Tên giọng đọc (Edge-TTS) hoặc mã ngôn ngữ (XTTSv2: `vi`, `en`). | Lấy từ `configs/default.json` |
| `--ref_audio` | Không | Đường dẫn tệp âm thanh mẫu `.wav` (Bắt buộc với XTTSv2, VieNeu). | Lấy từ `configs/default.json` |
| `--config` | Không | Đường dẫn đến tệp cấu hình JSON tùy chỉnh. | `configs/default.json` |
| `--output_dir` | Không | Thư mục đầu ra chứa tệp âm thanh kết quả. | `data/outputs` |
| `--phonemize` | `--no-phonemize` | Bật/tắt chuyển đổi văn bản sang phiên âm quốc tế IPA (XTTSv2). | `false` |
| `--normalize` | `--no-normalize` | Bật/tắt chuẩn hóa âm lượng theo tiêu chuẩn LUFS. | `true` |
| `--target_lufs` | Không | Mức âm lượng LUFS mục tiêu. | `-14.0` |
| `--fade_in` | Không | Thời gian hiệu ứng Fade-in ở đầu tệp (giây). | `0.1` |
| `--fade_out` | Không | Thời gian hiệu ứng Fade-out ở cuối tệp (giây). | `0.1` |
| `--silence_duration`| Không | Khoảng nghỉ lặng (giây) giữa các phân đoạn câu. | `0.3` |
| `--use_cache` | `--no-use_cache` | Bật/tắt bộ nhớ đệm Semantic Cache cho các phân đoạn âm thanh. | `false` |
| `--cache_threshold`| Không | Ngưỡng cosine similarity tương đồng văn bản để kích hoạt cache hit. | `0.95` |
| `--vieneu_mode` | Không | Chọn chế độ của động cơ VieNeu-TTS (`v3turbo`, `standard`). | `v3turbo` |
| `--vieneu_emotion`| Không | Chọn kiểu đọc/cảm xúc của động cơ VieNeu (`natural`, `storytelling`). | `natural` |
| `--vieneu_temp` | Không | Tham số Temperature điều chỉnh độ ngẫu nhiên của VieNeu. | `0.3` |
| `--ref_text` | Không | Transcript văn bản đi kèm file âm thanh mẫu (VieNeu/XTTS). | `None` |
| `--spice_text` | `--no-spice_text` | Bật/tắt viết lại văn bản với cảm xúc bằng Local LLM. | `false` |
| `--llm_model` | Không | Đường dẫn tới tệp Local GGUF LLM. | `None` |
| `--rvc_model` | Không | Đường dẫn tới tệp RVC `.pth` phục vụ đổi giọng (Voice-to-Voice). | `None` |
| `--rvc_index` | Không | Đường dẫn tới tệp RVC `.index` bổ trợ cải thiện giọng (tùy chọn). | `None` |
| `--rvc_pitch` | Không | Dịch chuyển cao độ tông giọng (pitch shift semitones: ví dụ +12 lên Nữ, -12 xuống Nam). | `0` |


---

### Các Ví Dụ Lệnh Thực Tế

#### A. Sử dụng Edge-TTS (Trực tuyến, miễn phí, chất lượng cao)
* Chạy với cấu hình mặc định (Giọng Nam Minh):
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine edge
  ```
* Chạy với giọng nữ `vi-VN-HoaiMyNeural` và tăng tốc độ đọc lên `1.15` lần:
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine edge --voice vi-VN-HoaiMyNeural --speed 1.15
  ```

#### B. Sử dụng Piper TTS (Ngoại tuyến, siêu nhẹ, chạy mượt trên CPU)
* Chạy với mô hình local `vi_VN-vais1000-medium.onnx`:
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine piper --model models/piper/vi_VN-vais1000-medium.onnx
  ```

#### C. Sử dụng XTTSv2 Voice Cloning (Ngoại tuyến, sao chép giọng nói mẫu)
* Nhân bản giọng đọc từ tệp mẫu `ref_voice.wav` bằng ngôn ngữ tiếng Việt:
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine clone --model models/xtts_v2 --ref_audio data/voices/ref_voice.wav --voice vi
  ```

#### D. Xử lý hàng loạt (Batch Processing)
* Quét toàn bộ file `.md` trong thư mục đầu vào và xuất ra các thư mục tương ứng trong thư mục đầu ra:
  ```powershell
  python main.py --input_dir data/inputs --engine edge --output_dir data/outputs
  ```

#### E. Sử dụng tệp cấu hình tùy biến
* Bạn có thể lưu cấu hình ưa thích vào `configs/default.json` hoặc tạo file cấu hình mới và gọi qua tham số `--config`:
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --config configs/default.json
  ```

#### F. Chạy Toàn Bộ Pipeline Spice & Clone (AI Spice + Edge-TTS + RVC)
* Chạy chuỗi đầy đủ: viết lại văn bản bằng GGUF LLM cục bộ, sinh âm thanh qua Edge-TTS (giọng Việt Nam), sau đó thực hiện chuyển đổi giọng nói Voice-to-Voice thông qua RVC:
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine edge --voice vi-VN-NamMinhNeural --spice_text --llm_model models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf --rvc_model models/rvc/adam.pth --rvc_pitch 0
  ```
  *(Lưu ý: Đối số `--rvc_pitch` dùng để tinh chỉnh cao độ của giọng nói: nam sang nữ đặt thành `12` hoặc ngược lại đặt thành `-12`)*

#### G. Sử dụng RVC Standalone (Đổi giọng trực tiếp cho tệp ghi âm sẵn có)
* Chuyển đổi giọng của một tệp âm thanh ghi âm có sẵn bằng mô hình RVC:
  ```powershell
  python main.py --input data/recordings/my_voice.wav --engine rvc --rvc_model models/rvc/ElevenLabs_Adam_FR.pth --rvc_index models/rvc/added_IVF4988_Flat_nprobe_1_ElevenLabs_Adam_FR_v2.index --rvc_pitch 0
  ```

#### H. Sử dụng VieNeu-TTS (Đọc giọng Việt offline tự nhiên)
* Sinh giọng VieNeu ngoại tuyến tốc độ cao (`v3turbo`) với biểu cảm kể chuyện (`storytelling`):
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine vieneu --vieneu_mode v3turbo --vieneu_emotion storytelling
  ```

#### I. Sử dụng Kokoro-Vietnamese (Offline PyTorch nhẹ)
* Sinh giọng đọc bằng mô hình Kokoro tiếng Việt với giọng đọc `diem_trinh`:
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine kokoro --voice diem_trinh
  ```

#### J. Sử dụng Valtec-TTS (Offline VITS Model)
* Sinh giọng đọc qua mô hình VITS Việt của Valtec với speaker `NF` (Nữ phía Bắc):
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine valtec --voice NF
  ```

#### K. Tăng tốc bằng Semantic Cache
* Sử dụng Semantic Cache để lưu lại các phân đoạn đã sinh, giúp các lần chạy tiếp theo của các câu tương đồng diễn ra ngay lập tức mà không cần sinh lại từ mô hình AI:
  ```powershell
  python main.py --input data/inputs/cau_chuyen_buoi_sang.md --engine edge --use_cache --cache_threshold 0.95
  ```

---

## ⚙️ Chi Tiết Hậu Xử Lý Âm Thanh & Phiên Âm

Để đạt chất lượng âm thanh chuyên nghiệp như phòng thu phát thanh, hệ thống tích hợp sẵn các công cụ bổ trợ trong [src/utils/](src/utils):

### 1. Chuẩn Hóa Âm Lượng LUFS (`--normalize`)
Hệ thống sử dụng thư viện `pyloudnorm` để đo lường độ lớn âm thanh tích hợp (Integrated Loudness) theo chuẩn **ITU-R BS.1770**. 
* Mức mặc định là `-14.0 LUFS` (đây là chuẩn âm lượng của các nền tảng phát trực tuyến như Spotify, YouTube).
* **Cơ chế dự phòng (Fallback):** Nếu máy bạn chưa cài đặt `pyloudnorm`, hệ thống sẽ tự động chuyển đổi sang Peak Normalization (đưa đỉnh biên độ âm thanh về mức `0.95` để tránh méo tiếng/clipping).

### 2. Fade-in & Fade-out (`--fade_in` / `--fade_out`)
Áp dụng bộ lọc giảm âm lượng tuyến tính ở đầu và cuối tệp âm thanh (mặc định `0.1 giây`). Điều này giúp triệt tiêu tiếng bụp phát thanh đột ngột khi thiết bị bắt đầu phát nhạc.

### 3. Ghép Đoạn kèm Khoảng Lặng (`--silence_duration`)
Vì các đoạn văn bản dài dễ gây tràn bộ nhớ GPU, văn bản được chia nhỏ thành các câu (dưới 30 từ). Sau khi sinh xong từng câu, hệ thống ghép nối các mảng numpy lại và chèn một khoảng lặng tĩnh dài `0.3 giây` (hoặc tùy chỉnh) giữa các câu để giọng đọc nghe tự nhiên, có nhịp thở.

### 4. Phiên Âm IPA Tiếng Việt (`--phonemize`)
Khi sử dụng XTTSv2, bộ phiên âm IPA tiếng Việt (sử dụng thư viện `viphoneme`) giúp chuyển đổi các từ tiếng Việt sang các ký tự ngữ âm quốc tế tương ứng trước khi truyền vào mạng neural của XTTS. Việc này cải thiện đáng kể độ chính xác phát âm và giảm thiểu hiện tượng nuốt từ hoặc nói lắp của mô hình clone.

---

## 🖥️ Chẩn Đoán GPU & Cấu Hình NVIDIA CUDA

Để chạy mô hình XTTSv2 ổn định, bạn cần chạy trên GPU NVIDIA có hỗ trợ CUDA.

### Chạy Công Cụ Chẩn Đoán
Hệ thống đi kèm một file chẩn đoán hệ thống tự động:
```powershell
python check_gpu.py
```
Bộ chẩn đoán sẽ kiểm tra và hiển thị các thông tin:
1. Phiên bản PyTorch đang cài đặt.
2. CUDA có khả dụng hay không.
3. Phiên bản Driver CUDA và cuDNN.
4. Danh sách các card đồ họa NVIDIA đang kết nối cùng dung lượng VRAM thực tế.
5. Kiểm tra tính năng **SDPA (Scaled Dot-Product Attention)** trực tiếp trên GPU để đảm bảo hiệu suất tính toán tốt nhất.

### Khắc phục lỗi CUDA chưa khả dụng
Nếu chạy `check_gpu.py` báo CUDA không khả dụng, điều này có nghĩa là bạn đang cài đặt phiên bản PyTorch dành cho CPU. Hãy chạy lệnh sau để cài đặt phiên bản PyTorch CUDA tương thích:
* **Đối với card RTX 30-Series và 40-Series (Tiêu chuẩn):**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
* **Đối với card RTX 50-Series (Kiến trúc Blackwell mới, ví dụ RTX 5060):** Bắt buộc sử dụng CUDA 12.8 để tránh lỗi crash `CUDA error: no kernel image is available`:
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

### Bộ tệp tin kích hoạt nhanh (.bat / .sh) tiện lợi
Dự án đi kèm các tập lệnh khởi chạy nhanh được thiết kế đặc biệt cho cả Windows (`.bat`) và macOS/Linux (`.sh`):
* **Khởi chạy Web UI**: Chạy `chay_giao_dien.bat` (Windows) hoặc `./chay_giao_dien.sh` (macOS/Linux) để khởi động máy chủ Web UI cục bộ.
* **Chẩn đoán GPU**: Chạy `kiem_tra_gpu.bat` (Windows) hoặc `./kiem_tra_gpu.sh` (macOS/Linux) để chạy nhanh báo cáo chẩn đoán GPU/CUDA.
* **Chạy kiểm thử**: Chạy `chay_kiem_thu.bat` (Windows) hoặc `./chay_kiem_thu.sh` (macOS/Linux) để chạy nhanh bộ kiểm thử tích hợp (tests suite).

---

## 🔌 Hướng Dẫn Mở Rộng Động Cơ Mới (Developer Guide)

Nhờ áp dụng nguyên lý **Clean Architecture**, bạn có thể dễ dàng bổ sung động cơ TTS mới (ví dụ: Google TTS, Coqui TTS khác, VibeVoice...) mà không ảnh hưởng tới logic xử lý văn bản hay hậu xử lý âm thanh trong `main.py`.

### Các bước thêm động cơ mới:

1. **Tạo file adapter mới** trong thư mục [src/engines/](src/engines) (ví dụ: `google.py`).
2. **Kế thừa lớp cha** `BaseTTSEngine` và ghi đè phương thức `generate()`:
   ```python
   # src/engines/google.py
   from src.engines.base import BaseTTSEngine

   class GoogleEngine(BaseTTSEngine):
       def __init__(self, key: str = None):
           self.key = key
           # Khởi tạo API Client của bạn tại đây...

       def generate(self, text: str, output_path: str, **kwargs) -> bool:
           try:
               # 1. Thực hiện sinh âm thanh từ text
               # 2. Ghi kết quả ra tệp tin tại output_path
               # 3. Trả về True nếu thành công, False nếu lỗi
               return True
           except Exception as e:
               print(f"Google TTS failed: {e}")
               return False
   ```
3. **Đăng ký động cơ mới** trong [main.py](main.py#L614-L630):
   ```python
   elif args.engine == "google":
       from src.engines.google import GoogleEngine
       engine = GoogleEngine(args.model)
   ```

---

## 🔍 Lỗi Thường Gặp & Cách Khắc Phục (Troubleshooting)

### 1. Lỗi `ImportError: DLL load failed` hoặc lỗi FFmpeg/torchaudio
* **Nguyên nhân:** Trên Windows, việc cài đặt `torchaudio` thường yêu cầu các thư viện C++ bổ sung hoặc FFmpeg DLL để giải mã âm thanh WAV.
* **Cách khắc phục:** Khung làm việc AIVoice đã tích hợp sẵn cơ chế **monkeypatch** tự động trong [src/engines/clone.py](src/engines/clone.py). Lớp này ghi đè hàm `torchaudio.load` và `torchaudio.save` bằng thư viện `soundfile` thuần Python. Nhờ đó, bạn **không cần cài đặt FFmpeg** hay cấu hình biến môi trường phức tạp mà hệ thống vẫn nhân bản giọng nói hoàn hảo.

### 2. Lỗi thiếu `Microsoft Visual C++ 14.0` khi cài đặt `coqui-tts`
* **Nguyên nhân:** Thư viện XTTSv2 chứa một số mô-đun yêu cầu biên dịch C++ (ví dụ: `vits` tokenizer hoặc `mecab`).
* **Cách khắc phục:** Tải xuống và cài đặt [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Trong trình cài đặt, hãy tích chọn mục **C++ build tools** (hoặc "Desktop development with C++") rồi tiến hành cài đặt và khởi động lại máy.

### 3. Lỗi tải mô hình bị treo hoặc tải chậm (Hugging Face)
* **Nguyên nhân:** Mạng kết nối trực tiếp đến Hugging Face đôi khi bị bóp băng thông hoặc tường lửa chặn tải file lớn (>2GB).
* **Cách khắc phục:** Sử dụng script tải chuyên dụng được cung cấp sẵn:
  ```powershell
  python download_models.py --engine all
  ```
  Công cụ này sử dụng gói `tqdm` để hiển thị thanh tiến trình tải xuống chi tiết và lưu trực tiếp vào thư mục mô hình cục bộ, đảm bảo tính ổn định tối đa.

### 4. Lỗi hiển thị hoặc lỗi chữ tiếng Việt có dấu khi gọi Piper (Subprocess)
* **Nguyên nhân:** Bảng mã mặc định của Command Prompt Windows (cmd.exe) thường là CP932 hoặc CP1252, không hỗ trợ tốt UTF-8.
* **Cách khắc phục:**
  * Hệ thống đã tự động cấu hình biến môi trường `PYTHONIOENCODING=utf-8` và thiết lập `encoding='utf-8'` khi gọi Piper dưới dạng tiến trình con (subprocess).
  * Khuyên dùng **PowerShell** hoặc **Windows Terminal** hiện đại thay cho `cmd.exe` truyền thống để hiển thị đúng ký tự tiếng Việt.

---

## 🧪 Hệ Thống Kiểm Thử Tự Động

Để đảm bảo các thay đổi mã nguồn không phá vỡ cấu trúc hệ thống, chạy kịch bản kiểm thử tích hợp:
```powershell
python tests/run_tests.py
```
Kịch bản này chạy qua 10 kịch bản kiểm tra toàn diện, bao gồm cả kiểm tra lỗi đầu vào âm bản và kiểm tra chất lượng file WAV đầu ra. Các tệp tin tạm thời và kết quả sẽ được ghi độc lập tại thư mục `tests/test_data/outputs/` để giữ cho môi trường làm việc của bạn luôn ngăn nắp.
