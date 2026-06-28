# 🎙️ BÁO CÁO TOÀN DIỆN DỰ ÁN AIVOICE (Dành Cho Nhà Phát Triển & AI Agent)

Tài liệu này tổng hợp toàn bộ thông tin chi tiết của dự án **AIVoice** (Khung Chuyển Đổi Văn Bản Thành Giọng Nói Tiếng Việt Đa Động Cơ). Tài liệu được cấu trúc chi tiết từ mức độ thiết kế kiến trúc lõi, giải thích từng tệp tin nguồn, các thuật toán tối ưu hóa phần cứng, các bản vá lỗi Windows/GPU đặc thù, đến hướng dẫn từng bước (step-by-step) để một lập trình viên mới có thể hiểu và tham gia phát triển dự án ngay lập tức mà không cần đọc lại tài liệu nào khác.

---

## 1. Tổng Quan Dự Án & Định Hướng Kiến Trúc

### 1.1. Mục tiêu dự án
* **Tên dự án:** AIVoice - Single-Speaker Local Vietnamese TTS Framework.
* **Định hướng phát triển:** Trở thành một framework chuyển đổi văn bản thành giọng nói (TTS) tiếng Việt hoạt động cục bộ (Offline-First), có tính mô-đun hóa cao, dễ dàng mở rộng và tích hợp thêm các công nghệ mới.
* **Các động cơ hỗ trợ:**
  1. **Edge-TTS (Online):** Sử dụng API giọng đọc mạng neural của Microsoft Edge (tốc độ cao, giọng nói tự nhiên, không tốn tài nguyên máy tính).
  2. **Piper (Offline):** Bộ tổng hợp giọng nói cực nhẹ chạy qua mô hình ONNX Runtime, lý tưởng cho CPU hoặc GPU cấp thấp.
  3. **XTTSv2 Voice Cloning (Offline):** Nhân bản giọng nói bất kỳ từ một file âm thanh mẫu (~6 giây) cục bộ qua mô hình Coqui XTTSv2.
  4. **RVC Voice Conversion (Offline):** Chuyển đổi giọng nói (Voice-to-Voice) thông qua Retrieval-based Voice Conversion như một bước hậu xử lý hoặc bộ lọc độc lập.
  5. **VieNeu-TTS (Offline/Online):** Động cơ sinh giọng nói tiếng Việt tự nhiên chất lượng cao chạy offline với nhiều chế độ và biểu cảm.
  6. **Kokoro-Vietnamese (Offline):** Động cơ sinh giọng nói tiếng Việt chạy trên PyTorch siêu nhẹ với nhiều giọng đọc chất lượng cao.
  7. **Valtec-TTS (Offline):** Động cơ sinh giọng nói tiếng Việt dựa trên VITS chạy offline.
  8. **Local AI Spice (Offline):** Tích hợp GGUF LLM cục bộ (thông qua `llama-cpp-python`) để viết lại văn bản đầu vào sinh động trước khi đọc.

### 1.2. Nguyên lý thiết kế (Clean Architecture)
Dự án áp dụng nguyên lý **Tách biệt Mối quan tâm (Separation of Concerns)**:
* **Giao diện người dùng (UI/CLI)** tách biệt hoàn toàn khỏi logic xử lý mô hình AI.
* **Lớp Adapters (Engines)** kế thừa một giao diện chung để có thể thay thế hoặc bổ sung các công nghệ TTS mới mà không ảnh hưởng tới luồng xử lý chính.
* **Lớp Tiện ích (Utils)** quản lý độc lập các tác vụ phụ trợ như tiền xử lý văn bản, hậu xử lý âm thanh, bộ nhớ đệm (Semantic Cache), và chuyển đổi IPA.

---

## 2. Bản Đồ Cấu Trúc Thư Mục (Codebase Structure)

Dưới đây là sơ đồ chi tiết cấu trúc thư mục của dự án và vai trò của từng tệp tin:

```text
AIVoice/
│
├── configs/               # [1. CẤU HÌNH] Chứa các tệp thiết lập tham số hệ thống
│   └── default.json       # Tệp cấu hình mặc định (tốc độ, âm lượng, đường dẫn model...)
│
├── data/                  # [2. DỮ LIỆU NGƯỜI DÙNG] (Được đưa vào .gitignore)
│   ├── inputs/            # Thư mục chứa tài liệu văn bản (.md, .txt) cần đọc
│   ├── outputs/           # Thư mục tự động sinh ra tệp âm thanh kết quả (.wav)
│   └── voices/            # Chứa các file giọng nói mẫu (ref_voice.wav) phục vụ nhân bản giọng
│
├── models/                # [3. TRỌNG SỐ MÔ HÌNH AI CỤC BỘ] (Bị bỏ qua bởi Git)
│   ├── piper/             # Chứa tệp vi_VN-vais1000-medium.onnx và tệp cấu hình JSON
│   ├── xtts_v2/           # Chứa mô hình XTTSv2 fine-tuned cho tiếng Việt (model.pth, config.json...)
│   ├── llm/               # Chứa các file mô hình GGUF LLM cục bộ (e.g. Qwen2.5-1.5B)
│   └── rvc/               # Chứa mô hình RVC (.pth) và file chỉ mục (.index) dùng để chuyển giọng
│
├── src/                   # [4. MÃ NGUỒN LÕI]
│   ├── engines/           # Các lớp động cơ TTS (Adapters)
│   │   ├── base.py        # Giao diện lớp cha trừu tượng (BaseTTSEngine)
│   │   ├── edge.py        # Động cơ Microsoft Edge TTS Online
│   │   ├── piper.py       # Động cơ Piper TTS ONNX Offline
│   │   ├── clone.py       # Động cơ nhân bản giọng Coqui XTTSv2 Offline
│   │   ├── rvc_engine.py  # Động cơ đổi giọng nói Voice-to-Voice RVC Offline
│   │   ├── kokoro.py      # Động cơ Kokoro-Vietnamese Offline
│   │   ├── valtec.py      # Động cơ Valtec-TTS Offline
│   │   └── vieneu.py      # Động cơ VieNeu-TTS Offline/Online
│   │
│   └── utils/             # Các thư viện bổ trợ
│       ├── audio.py       # Ghép nối file âm thanh, fade-in/out, chuẩn hóa LUFS, cắt nhỏ file cho RVC
│       ├── text.py        # Làm sạch cú pháp markdown và thuật toán phân chia đoạn văn (chunking)
│       ├── cache.py       # Bộ quản lý bộ nhớ đệm thông minh Semantic Cache
│       ├── phoneme.py     # Phiên âm từ tiếng Việt sang IPA (hỗ trợ XTTSv2)
│       └── local_ai_spice.py # Viết lại văn bản đầu vào sinh động qua mô hình LLM cục bộ
│
├── tests/                 # [5. HỆ THỐNG KIỂM THỬ ĐỘC LẬP]
│   ├── test_data/         # Dữ liệu đầu vào/đầu ra riêng phục vụ kiểm thử
│   └── run_tests.py       # Tập lệnh chạy tự động các kịch bản kiểm thử toàn diện
│
├── check_gpu.py           # Công cụ chẩn đoán khả năng tương thích GPU/CUDA và kiểm tra tính năng SDPA
├── download_models.py     # Tập lệnh tự động tải xuống các mô hình (Piper & XTTSv2) từ Hugging Face
├── main.py                # Điểm khởi chạy (Entry Point) của CLI và Wizard tương tác từng bước
├── requirements.txt       # Danh sách thư viện và dependencies cài đặt qua pip
└── setup.bat              # Tập lệnh tự động thiết lập môi trường Python ảo (.venv) trên Windows
```

---

## 3. Mô-tả Chi Tiết Mã Nguồn Từng File Code

### 3.1. Các lớp Động cơ (src/engines/)

#### 1. [src/engines/base.py](file:///f:/programfiles/AIVoice/src/engines/base.py)
* **Nhiệm vụ:** Định nghĩa lớp cha trừu tượng `BaseTTSEngine` kế thừa từ `abc.ABC`.
* **Logic lõi:** Yêu cầu mọi động cơ kế thừa phải triển khai phương thức:
  ```python
  @abstractmethod
  def generate(self, text: str, output_path: str, **kwargs) -> bool:
      pass
  ```

#### 2. [src/engines/edge.py](file:///f:/programfiles/AIVoice/src/engines/edge.py)
* **Nhiệm vụ:** Gọi API chuyển đổi văn bản của Microsoft Edge thông qua thư viện `edge-tts`.
* **Logic lõi:** Chạy bất đồng bộ (`asyncio`) bằng cách tự thiết lập một event loop cục bộ.

#### 3. [src/engines/piper.py](file:///f:/programfiles/AIVoice/src/engines/piper.py)
* **Nhiệm vụ:** Cấu hình và chạy mô hình ONNX Runtime của Piper TTS.
* **Logic lõi:** Nhận tệp `.onnx` và chạy suy luận (inference). Hỗ trợ tăng tốc phần cứng thông qua `onnxruntime-gpu` nếu có GPU NVIDIA tương thích.

#### 4. [src/engines/clone.py](file:///f:/programfiles/AIVoice/src/engines/clone.py)
* **Nhiệm vụ:** Đóng gói mô hình nhân bản giọng nói XTTSv2 (Coqui TTS) chạy Offline.
* **Các cơ chế đặc biệt:**
  * **Vá Torchaudio:** Ghi đè `torchaudio.load` và `torchaudio.save` để đọc/ghi tệp âm thanh bằng thư viện `soundfile` thuần Python, giải quyết triệt để lỗi thiếu thư viện DLL FFmpeg trên Windows.
  * **Bộ đệm thông số giọng mẫu (Latents Caching):** Lưu trữ kết quả trích xuất đặc trưng giọng đọc (`gpt_cond_latent`, `speaker_embedding`) từ file mẫu `ref_voice.wav`. Khi chuyển đổi hàng loạt câu văn dài, nếu file giọng mẫu không đổi, lớp sẽ tái sử dụng đặc trưng này để tăng tốc suy luận.
  * **Mạng lưới bảo vệ ký tự tiếng Việt:** XTTSv2 tiếng Việt bị giới hạn cứng 250 ký tự. Nếu vượt quá, mô hình sẽ nạp sai chỉ mục và gây crash CUDA. Hàm `generate` tự động kiểm tra, nếu văn bản dài hơn 248 ký tự thì tiến hành cắt ngắn về 240 ký tự trước khi truyền cho mô hình, đảm bảo tính liên tục của hệ thống.

---

### 3.2. Thư mục Tiện ích (src/utils/)

#### 1. [src/utils/text.py](file:///f:/programfiles/AIVoice/src/utils/text.py)
* **Nhiệm vụ:** Làm sạch cú pháp markdown và bẻ văn bản dài thành các phân đoạn nhỏ.
* **Thuật toán `chunk_text`:**
  * Tách văn bản thô theo dấu câu (`.`, `?`, `!`, `;`).
  * Kiểm tra độ dài từng phân đoạn. Nếu câu vượt quá `max_words=50` từ hoặc **`max_chars=240` ký tự**, hàm sẽ bẻ nhỏ câu tại vị trí dấu phẩy hoặc khoảng trắng gần nhất.
  * Ghép các câu cực ngắn lại với nhau (nếu tổng số từ vẫn $\le$ `max_words` và tổng số ký tự $\le$ `max_chars`) để giọng đọc trôi chảy, có ngữ điệu tự nhiên.

#### 2. [src/utils/audio.py](file:///f:/programfiles/AIVoice/src/utils/audio.py)
* **Nhiệm vụ:** Quản lý hậu xử lý âm thanh.
* **Logic lõi:**
  * `concatenate_wavs`: Đọc các tệp âm thanh phân đoạn tạm thời và ghép chúng lại thành một file WAV duy nhất, chèn khoảng lặng (`silence_duration`) giữa các đoạn.
  * `apply_audio_processing`: Áp dụng chuẩn hóa âm lượng theo chuẩn phát thanh LUFS (`pyloudnorm`), tăng tốc hoặc giảm tốc độ đọc (`librosa`), và tạo hiệu ứng bo viền âm lượng fade-in / fade-out ở đầu và cuối tệp để tránh tiếng click âm thanh đột ngột.

#### 3. [src/utils/phoneme.py](file:///f:/programfiles/AIVoice/src/utils/phoneme.py)
* **Nhiệm vụ:** Chuẩn hóa và phiên âm văn bản sang ký tự IPA tiếng Việt.
* **Logic lõi:** Gọi thư viện `vinorm` và `viphoneme`. Ghi đè hàm `vinorm.TTSnorm` để tránh gọi các tệp nhị phân Linux trên Windows.

#### 4. [src/utils/local_ai_spice.py](file:///f:/programfiles/AIVoice/src/utils/local_ai_spice.py)
* **Nhiệm vụ:** Sử dụng LLM GGUF cục bộ để viết lại văn bản.
* **Logic lõi:** Gọi `llama_cpp.Llama`, thiết lập tham số prompt hệ thống và thực hiện giải phóng tài nguyên CUDA ngay lập tức sau khi sinh văn bản.

---

## 4. Các Giải Pháp Tối Ưu Hóa Bộ Nhớ VRAM (Tránh Lỗi Tràn VRAM OOM)

Đối với các GPU có bộ nhớ VRAM hạn chế (từ **6GB - 8GB** như NVIDIA RTX 3060, RTX 4060, hoặc RTX 5060 Laptop/Desktop), hệ thống áp dụng các tối ưu hóa phần cứng chuyên sâu:

1. **Hàng đợi đơn luồng xử lý GPU (Single-Worker Queue):**
   Trong `main.py`, khi sinh âm thanh bằng động cơ nhân bản giọng (`clone`), hệ thống sử dụng cấu hình `max_workers = 1`. GPU sẽ chỉ xử lý tuần tự từng phân đoạn câu văn bản một, loại bỏ hoàn toàn khả năng nhiều phân đoạn cùng nạp lên GPU gây tràn VRAM.
2. **Kích hoạt PyTorch SDPA (Scaled Dot-Product Attention):**
   Trong `src/engines/clone.py`, thay vì cài đặt thư viện `flash-attn` rất khó biên dịch trên Windows, hệ thống sử dụng context manager tích hợp sẵn của PyTorch:
   ```python
   sdpa_context = torch.backends.cuda.sdp_kernel(
       enable_flash=True, 
       enable_math=True, 
       enable_mem_efficient=True
   )
   ```
   Giúp tăng tốc độ suy luận của mô hình gần gấp đôi và tiết kiệm bộ nhớ đáng kể.
3. **Cơ chế thu hồi và dọn dẹp bộ nhớ RAM/VRAM:**
   Sau khi hoàn tất quá trình sinh giọng hoặc chuyển đổi giọng nói (RVC), hệ thống chủ động gọi trình dọn rác Python (`gc.collect()`) và giải phóng bộ đệm của GPU (`torch.cuda.empty_cache()`).
4. **Cắt nhỏ file âm thanh khi đổi giọng (RVC Slicing):**
   RVC tiêu thụ VRAM tăng theo cấp số nhân với độ dài tệp âm thanh đầu vào. Để xử lý các tệp âm thanh dài (từ vài phút đến vài tiếng), hệ thống sẽ tự động cắt âm thanh thành các đoạn ngắn tối đa `60 giây` trước khi đưa vào RVC, thực hiện chuyển đổi tuần tự rồi ghép nối lại.

---

## 5. Các Bản Vá Tương Thích Trên Môi Trường Windows (Windows Patches)

Dự án chứa nhiều bản vá động (monkey-patch) đặc thù để đảm bảo hoạt động trơn tru trên mọi hệ thống Windows:

1. **Hỗ trợ GPU Blackwell RTX 50-Series:**
   Thế hệ card RTX 5060/5070/5080/5090 sử dụng compute capability `sm_120` rất mới, yêu cầu PyTorch phiên bản mới được biên dịch kèm CUDA 12.6 hoặc CUDA 12.8 (ví dụ: `torch-2.11.0+cu128`).
2. **Bỏ qua lỗi nạp thư viện `torchcodec`:**
   Mô hình XTTSv2 trên PyTorch phiên bản mới cố gắng import `torchcodec`, thư viện này thường bị lỗi liên kết động (DLL Load Failed) với FFmpeg trên Windows. Hệ thống đã ghi đè hàm kiểm tra của Transformers và hàm lấy kích thước file của Coqui TTS để chuyển hướng sử dụng `torchaudio.info` làm fallback, bỏ qua hoàn toàn yêu cầu cài đặt `torchcodec`.
3. **Tìm kiếm tự động DLL CUDA của ONNX Runtime:**
   Trong `main.py`, khi chạy trên Windows, hệ thống sẽ tự động quét thư mục cài đặt của `torch` để tìm các thư viện DLL của CUDA (`cublas64_*.dll`, `cudart64_*.dll`, `libcudnn*.dll`) và thêm chúng vào đường dẫn tìm kiếm hệ thống (`os.add_dll_directory`). Người dùng **không cần thiết phải tự cài đặt CUDA Toolkit hoặc cuDNN** lên hệ điều hành của họ.
4. **Lỗi mã hóa Tkinter Subprocess (UTF-8 Output):**
   Trong `web_ui.py`, khi người dùng bấm chọn file/thư mục, hệ thống khởi chạy một tiến trình Python phụ chạy Tkinter. Để tránh việc in ký tự tiếng Việt có dấu ra màn hình CMD bị lỗi sập mã hóa `UnicodeEncodeError` (do Windows console mặc định sử dụng bảng mã `cp1252`), tiến trình phụ được cấu hình:
   ```python
   import sys; sys.stdout.reconfigure(encoding='utf-8')
   ```

---

## 6. Hướng Dẫn Từng Bước Dành Cho Nhà Phát Triển Mới

### 6.1. Cách tích hợp một động cơ TTS mới (Ví dụ: Google-TTS)

#### Bước 1: Tạo file adapter engine mới
Tạo tệp tin [src/engines/google_tts.py](file:///f:/programfiles/AIVoice/src/engines/google_tts.py):

```python
import os
from gtts import gTTS
from src.engines.base import BaseTTSEngine

class GoogleTTSEngine(BaseTTSEngine):
    """Google TTS Engine Adapter."""
    
    def __init__(self, default_lang: str = "vi"):
        self.default_lang = default_lang

    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        try:
            lang = kwargs.get("voice") or self.default_lang
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
            tts = gTTS(text=text, lang=lang)
            tts.save(output_path)
            return True
        except Exception as e:
            print(f"Lỗi Google TTS: {e}")
            return False
```

#### Bước 2: Đăng ký Engine mới trong CLI
Mở file [main.py](file:///f:/programfiles/AIVoice/main.py):
1. Thêm import:
   ```python
   from src.engines.google_tts import GoogleTTSEngine
   ```
2. Thêm logic khởi tạo trong hàm xử lý (nơi kiểm tra `args.engine`):
   ```python
   elif args.engine == "google":
       engine = GoogleTTSEngine(default_lang="vi")
   ```
3. Bổ sung `"google"` vào tùy chọn `choices` của tham số `--engine` trong hàm `parse_args()` cuối file:
   ```python
   parser.add_argument("--engine", choices=["edge", "piper", "clone", "google"], ...)
   ```

#### Bước 3: Đăng ký Engine mới trên Web UI
Mở file [templates/index.html](file:///f:/programfiles/AIVoice/templates/index.html), tìm đến thẻ `<select id="engine">` và thêm tùy chọn:
```html
<option value="google">Google TTS (Online)</option>
```

---

### 6.2. Cách viết kịch bản kiểm thử mới cho động cơ mới

Mở file [tests/run_tests.py](file:///f:/programfiles/AIVoice/tests/run_tests.py), tìm mảng `test_cases` và thêm kịch bản test mới:

```python
        {
            "name": "test_google_default",
            "desc": "Google TTS online engine integration test",
            "args": ["main.py", "--input", "tests/test_data/inputs/test_vi.md", "--engine", "google"],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        }
```

Chạy bộ kiểm thử tự động để xác nhận hoạt động:
```powershell
.\.venv\Scripts\python.exe tests/run_tests.py
```

---

## 7. Quy Trình Cài Đặt Hệ Thống Mới từ A-Z

### 7.1. Các điều kiện chuẩn bị trước (Prerequisites)
1. Cài đặt **Python 3.11.x** (khuyên dùng bản 3.11.9, nhớ tích chọn *"Add Python to PATH"* khi cài đặt).
2. Cài đặt **Microsoft C++ Build Tools** (để biên dịch các phần mở rộng C++ của `coqui-tts`, `llama-cpp-python` và `fairseq`):
   * Tải bộ cài installer từ trang chủ Microsoft Visual Studio.
   * Khởi chạy, tích chọn workload **"Desktop development with C++"** và tiến hành cài đặt. Cài xong cần khởi động lại máy tính.
3. Cài đặt **Git for Windows** (để hỗ trợ pip cài đặt các thư viện ngữ âm trực tiếp từ kho mã nguồn Github).

### 7.2. Các bước cài đặt tự động
1. Bấm đúp tệp **`setup.bat`** ở thư mục dự án. Script sẽ tự động:
   * Tạo môi trường ảo `.venv` nếu chưa có.
   * Cài đặt phiên bản `pip==24.0`.
   * Cài đặt toàn bộ dependencies được khai báo trong `requirements.txt`.
   * Cài đặt các thư viện chuyển đổi IPA từ Github (`viphoneme`, `vinorm`).
   * Tự động tải xuống các trọng số mô hình AI (Piper & XTTSv2) từ Hugging Face về thư mục `models/`.
2. Để chạy suy luận gia tốc phần cứng trên GPU NVIDIA (Ví dụ: RTX 5060), anh chạy lệnh cài đặt PyTorch phiên bản hỗ trợ CUDA 12.8 từ file nhị phân đã tải về:
   ```powershell
   .\.venv\Scripts\pip install "đường_dẫn_đến_file_torch_cu128.whl"
   ```

---

## 8. Bảng Tra Cứu Lệnh Vận Hành Nhanh (Cheatsheet)

* **Khởi chạy giao diện Web:**
  ```powershell
  .\chay_giao_dien.bat
  ```
* **Chạy giao diện dòng lệnh CLI (Nhân bản giọng nói XTTSv2 bằng GPU):**
  ```powershell
  .\.venv\Scripts\python.exe main.py --input data/inputs/test.txt --engine clone --ref_audio data/voices/ref_voice.wav --voice vi
  ```
* **Chạy giao diện dòng lệnh CLI (Động cơ Piper TTS ONNX):**
  ```powershell
  .\.venv\Scripts\python.exe main.py --input data/inputs/test.txt --engine piper --model models/piper/vi_VN-vais1000-medium.onnx
  ```
* **Chạy bộ kiểm thử tự động hệ thống:**
  ```powershell
  .\.venv\Scripts\python.exe tests/run_tests.py
  ```
* **Chạy chẩn đoán tương thích phần cứng GPU:**
  ```powershell
  .\.venv\Scripts\python.exe check_gpu.py
  ```
