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
6. [🔌 Hướng Dẫn Mở Rộng Động Cơ Mới (Developer Guide)](#-hướng-dẫn-mở-rộng-động-cơ-mới-developer-guide)
7. [🧪 Hệ Thống Kiểm Thử Tự Động](#-hệ-thống-kiểm-thử-tự-động)

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
│   └── xtts_v2/           # Chứa mô hình XTTSv2 fine-tuned và các file phụ trợ
│
├── src/                   # [4. LÕI LOGIC] Nơi chứa toàn bộ mã nguồn xử lý logic
│   ├── engines/           # Các Động Cơ TTS bọc thành lớp độc lập
│   │   ├── base.py        # Giao diện abstract lớp cha
│   │   ├── edge.py        # Bộ điều khiển Microsoft Edge Online TTS
│   │   ├── piper.py       # Bộ điều khiển Piper ONNX TTS Offline
│   │   └── clone.py       # Bộ điều khiển Coqui XTTSv2 Voice Cloning Offline
│   └── utils/             # Các thư viện bổ trợ
│       ├── audio.py       # Hậu xử lý âm lượng (LUFS), cắt ghép, fade-in/out
│       ├── text.py        # Làm sạch cú pháp markdown và phân tách câu (chunking)
│       └── phoneme.py     # Hỗ trợ chuyển đổi phiên âm sang IPA tiếng Việt
│
├── tests/                 # [5. BỘ KIỂM THỬ] Độc lập để xác thực hệ thống
│   ├── test_data/         # Dữ liệu phục vụ kiểm thử (inputs/outputs cô lập)
│   └── run_tests.py       # Script chạy toàn bộ kịch bản kiểm thử tự động
│
├── check_gpu.py           # Công cụ kiểm tra sức khỏe và chẩn đoán GPU/CUDA
├── download_models.py     # Script tải xuống mô hình tự động (Piper & XTTSv2)
├── main.py                # [ENTRY POINT] CLI điều khiển chính (Hỗ trợ Wizard tương tác)
├── requirements.txt       # Danh sách thư viện môi trường cần cài đặt
└── README.md              # Tài liệu hướng dẫn sử dụng này
```

---

## 🛠️ Hướng Dẫn Cài Đặt

### 1. Yêu Cầu Hệ Thống
* **Hệ điều hành:** Windows 10/11 (khuyên dùng có GPU NVIDIA VRAM >= 6GB để chạy sao chép giọng XTTSv2 nhanh).
* **Python:** Phiên bản 3.10 hoặc 3.11.

### 2. Tạo Môi Trường Ảo & Cài Đặt Thư Viện
Mở PowerShell trong thư mục dự án và chạy các lệnh sau:

```powershell
# Tạo môi trường ảo python
python -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\Activate.ps1

# Nâng cấp pip lên bản mới nhất
python -m pip install --upgrade pip

# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

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

Hệ thống cung cấp hai phương thức điều khiển: **Chế độ tương tác từng bước** và **Chế độ dòng lệnh CLI**.

### 1. Chế Độ Tương Tác (Wizard Mode - KHUYÊN DÙNG)
Nếu bạn không muốn nhớ các tham số phức tạp, chỉ cần gõ lệnh sau:
```powershell
python main.py
```
Hệ thống sẽ hiển thị một menu tương tác bằng tiếng Việt trên terminal để bạn chọn từng bước:
1. **Chọn Engine / Chế độ chạy:** `edge` (Microsoft), `piper` (Offline nhẹ), `clone` (XTTSv2 sao chép giọng), hoặc `batch` (Chạy hàng loạt).
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
| `--input` | Không | Đường dẫn đến tệp văn bản đầu vào `.md` cần chuyển đổi. | *Bắt buộc (hoặc dùng `--input_dir`)* |
| `--input_dir` | Không | Đường dẫn đến thư mục chứa nhiều tệp `.md` để chạy hàng loạt. | *Bắt buộc (nếu không dùng `--input`)* |
| `--engine` | Không | Chọn động cơ sinh giọng nói (`edge`, `piper`, `clone`). | Lấy từ `configs/default.json` |
| `--model` | Không | Đường dẫn tới tệp/thư mục chứa trọng số mô hình. | Lấy từ `configs/default.json` |
| `--speed` | Không | Tốc độ đọc của giọng nói (ví dụ: `1.0`, `1.2`, `0.85`). | `1.0` |
| `--voice` | Không | Tên giọng đọc (Edge-TTS) hoặc mã ngôn ngữ (XTTSv2: `vi`, `en`). | Lấy từ `configs/default.json` |
| `--ref_audio` | Không | Đường dẫn tệp âm thanh mẫu `.wav` (Bắt buộc với XTTSv2). | Lấy từ `configs/default.json` |
| `--config` | Không | Đường dẫn đến tệp cấu hình JSON tùy chỉnh. | `configs/default.json` |
| `--output_dir` | Không | Thư mục đầu ra chứa tệp âm thanh kết quả. | `data/outputs` |
| `--phonemize` | `--no-phonemize` | Bật/tắt chuyển đổi văn bản sang phiên âm quốc tế IPA (XTTSv2). | `false` |
| `--normalize` | `--no-normalize` | Bật/tắt chuẩn hóa âm lượng theo tiêu chuẩn LUFS. | `true` |
| `--target_lufs` | Không | Mức âm lượng LUFS mục tiêu. | `-14.0` |
| `--fade_in` | Không | Thời gian hiệu ứng Fade-in ở đầu tệp (giây). | `0.1` |
| `--fade_out` | Không | Thời gian hiệu ứng Fade-out ở cuối tệp (giây). | `0.1` |
| `--silence_duration`| Không | Khoảng nghỉ lặng (giây) giữa các phân đoạn câu. | `0.3` |

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

---

## ⚙️ Chi Tiết Hậu Xử Lý Âm Thanh & Phiên Âm

Để đạt chất lượng âm thanh chuyên nghiệp như phòng thu phát thanh, hệ thống tích hợp sẵn các công cụ bổ trợ trong [src/utils/](file:///F:/programfiles/AIVoice/src/utils):

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
Nếu chạy `check_gpu.py` báo CUDA không khả dụng, điều này có nghĩa là bạn đang cài đặt phiên bản PyTorch dành cho CPU. Hãy chạy lệnh sau để cài đặt phiên bản PyTorch CUDA 12.x:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## 🔌 Hướng Dẫn Mở Rộng Động Cơ Mới (Developer Guide)

Nhờ áp dụng nguyên lý **Clean Architecture**, bạn có thể dễ dàng bổ sung động cơ TTS mới (ví dụ: Google TTS, Coqui TTS khác, VibeVoice...) mà không ảnh hưởng tới logic xử lý văn bản hay hậu xử lý âm thanh trong `main.py`.

### Các bước thêm động cơ mới:

1. **Tạo file adapter mới** trong thư mục [src/engines/](file:///F:/programfiles/AIVoice/src/engines) (ví dụ: `google.py`).
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
3. **Đăng ký động cơ mới** trong [main.py](file:///F:/programfiles/AIVoice/main.py#L614-L630):
   ```python
   elif args.engine == "google":
       from src.engines.google import GoogleEngine
       engine = GoogleEngine(args.model)
   ```

---

## 🧪 Hệ Thống Kiểm Thử Tự Động

Để đảm bảo các thay đổi mã nguồn không phá vỡ cấu trúc hệ thống, chạy kịch bản kiểm thử tích hợp:
```powershell
python tests/run_tests.py
```
Kịch bản này chạy qua 10 kịch bản kiểm tra toàn diện, bao gồm cả kiểm tra lỗi đầu vào âm bản và kiểm tra chất lượng file WAV đầu ra. Các tệp tin tạm thời và kết quả sẽ được ghi độc lập tại thư mục `tests/test_data/outputs/` để giữ cho môi trường làm việc của bạn luôn ngăn nắp.
