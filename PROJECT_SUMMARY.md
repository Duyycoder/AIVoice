# 🎙️ BÁO CÁO TOÀN DIỆN DỰ ÁN AIVOICE (Dành Cho Claude Code Đọc & Đánh Giá)

Tài liệu này tổng hợp toàn bộ thông tin chi tiết của dự án **AIVoice** (Khung Chuyển Đổi Văn Bản Thành Giọng Nói Tiếng Việt Đa Động Cơ), từ kiến trúc hệ thống, chi tiết triển khai mã nguồn đến các kỹ thuật tối ưu hóa phần cứng được áp dụng để tránh lỗi tràn bộ nhớ (Out-Of-Memory - OOM) trên card đồ họa **NVIDIA GeForce RTX 5060 (8GB VRAM)**.

---

## 1. Tổng Quan Dự Án & Định Hướng
* **Tên dự án:** AIVoice - Single-Speaker Local Vietnamese TTS Framework (Extensible CLI-Driven).
* **Mục tiêu chính:** Cung cấp một framework chuyển đổi văn bản thành giọng nói (TTS) tiếng Việt hoạt động cục bộ ưu tiên ngoại tuyến (Offline-First), có tính mô-đun hóa cao, dễ dàng mở rộng và tích hợp thêm các công nghệ mới.
* **Các động cơ hỗ trợ:**
  1. **Edge-TTS (Online):** Sử dụng API giọng đọc mạng neural của Microsoft Edge (tốc độ cao, giọng nói tự nhiên, không tốn tài nguyên máy tính).
  2. **Piper (Offline):** Bộ tổng hợp giọng nói cực nhẹ chạy qua mô hình ONNX Runtime, lý tưởng cho CPU hoặc GPU cấp thấp.
  3. **XTTSv2 Voice Cloning (Offline):** Nhân bản giọng nói bất kỳ từ một file âm thanh mẫu (~6 giây) cục bộ qua mô hình Coqui XTTSv2.
  4. **RVC Voice Conversion (Offline):** Chuyển đổi giọng nói (Voice-to-Voice) thông qua Retrieval-based Voice Conversion như một bước hậu xử lý hoặc bộ lọc độc lập.
  5. **Local AI Spice (Offline):** Tích hợp GGUF LLM cục bộ (thông qua `llama-cpp-python`) để viết lại văn bản đầu vào thêm cảm xúc và yếu tố hài hước trước khi đọc.

---

## 2. Sơ Đồ Cấu Trúc Dự Án (Clean Architecture)

Dự án được cấu trúc rõ ràng theo nguyên lý tách biệt mối quan tâm (Separation of Concerns):

```text
AIVoice/
│
├── configs/               # [1. CẤU HÌNH] Chứa các tệp thiết lập tham số hệ thống
│   └── default.json       # Tệp cấu hình mặc định (tốc độ, âm lượng, đường dẫn model, RVC...)
│
├── data/                  # [2. DỮ LIỆU NGƯỜI DÙNG] Được đưa vào .gitignore trừ thư mục mẫu
│   ├── inputs/            # Thư mục chứa tài liệu văn bản (.md, .txt) cần đọc
│   ├── outputs/           # Thư mục tự động sinh ra tệp âm thanh kết quả (.wav)
│   └── voices/            # Chứa các file giọng nói mẫu (ref_voice.wav) phục vụ nhân bản giọng
│
├── models/                # [3. TRỌNG SỐ MÔ HÌNH AI CỤC BỘ] (Bị bỏ qua bởi Git)
│   ├── piper/             # Chứa tệp vi_VN-vais1000-medium.onnx và tệp cấu hình JSON
│   ├── xtts_v2/           # Chứa mô hình XTTSv2 fine-tuned cho tiếng Việt (dvae.pth, model.pth, vocab.json...)
│   ├── llm/               # Chứa các file mô hình GGUF LLM cục bộ (e.g. Qwen2.5-1.5B)
│   └── rvc/               # Chứa mô hình RVC (.pth) và file chỉ mục (.index) dùng để chuyển giọng
│
├── src/                   # [4. MÃ NGUỒN LÕI]
│   ├── engines/           # Các lớp động cơ TTS (Adapters)
│   │   ├── base.py        # Giao diện lớp cha trừu tượng (BaseTTSEngine)
│   │   ├── edge.py        # Động cơ Microsoft Edge TTS Online
│   │   ├── piper.py       # Động cơ Piper TTS ONNX Offline
│   │   ├── clone.py       # Động cơ nhân bản giọng Coqui XTTSv2 Offline
│   │   └── rvc_engine.py  # Động cơ đổi giọng nói Voice-to-Voice RVC Offline
│   │
│   └── utils/             # Các thư viện bổ trợ
│       ├── audio.py       # Ghép nối file âm thanh, fade-in/out, chuẩn hóa LUFS, cắt nhỏ file cho RVC
│       ├── text.py        # Làm sạch cú pháp markdown và thuật toán phân chia đoạn văn (chunking)
│       ├── phoneme.py     # Phiên âm từ tiếng Việt sang IPA (hỗ trợ XTTSv2)
│       └── local_ai_spice.py # Viết lại văn bản đầu vào sinh động qua mô hình LLM cục bộ
│
├── tests/                 # [5. HỆ THỐNG KIỂM THỬ ĐỘC LẬP]
│   ├── test_data/         # Dữ liệu đầu vào/đầu ra riêng phục vụ kiểm thử
│   └── run_tests.py       # Tập lệnh chạy tự động 11 kịch bản kiểm thử toàn diện
│
├── check_gpu.py           # Công cụ chẩn đoán khả năng tương thích GPU/CUDA và kiểm tra tính năng SDPA
├── download_models.py     # Tập lệnh tự động tải xuống các mô hình (Piper & XTTSv2) từ Hugging Face
├── main.py                # Điểm khởi chạy (Entry Point) của CLI và Wizard tương tác từng bước
├── web_ui.py              # Giao diện Web UI (Flask-based SPA với phong cách Glassmorphism)
├── requirements.txt       # Danh sách thư viện và dependencies cài đặt qua pip
├── plan.md                # Kế hoạch phát triển dự án và trạng thái hiện tại
├── HUONG_DAN_SU_DUNG.md   # Hướng dẫn sử dụng tiếng Việt chi tiết cho Web UI & sửa lỗi RVC/phomie
├── chay_giao_dien.bat     # Tập lệnh khởi chạy nhanh máy chủ Web UI cục bộ
├── chay_kiem_thu.bat      # Tập lệnh chạy nhanh bộ kiểm thử tích hợp (tests suite)
└── kiem_tra_gpu.bat       # Tập lệnh chạy nhanh chẩn đoán GPU/CUDA
```

---

## 3. Các Giải Giải Pháp Tối Ưu Hóa Tránh Tràn VRAM (OOM) Cho GPU 8GB (RTX 5060)

Để hệ thống hoạt động ổn định trên card đồ họa **RTX 5060 (8GB VRAM)** hoặc thấp hơn (thậm chí là **6GB VRAM**), một số chiến lược tối ưu hóa bộ nhớ chuyên sâu đã được tích hợp trực tiếp vào thiết kế dự án:

### 3.1. Thuật Toán Phân Đoạn Văn Bản (Text Chunking)
* **File nguồn:** [text.py](file:///f:/programfiles/AIVoice/src/utils/text.py)
* **Chi tiết:** Các mô hình học sâu như XTTSv2 tiêu thụ VRAM tăng theo cấp số nhân với chiều dài của đoạn văn bản đầu vào. Thuật toán `chunk_text` tự động làm sạch các cú pháp Markdown, tách các đoạn văn dài thành các câu ngắn dựa trên dấu kết thúc câu (`.`, `?`, `!`, `;`). Nếu câu vượt quá `30 từ` (mặc định), nó sẽ tiếp tục bị bẻ nhỏ tại dấu phẩy hoặc khoảng trắng gần nhất. Các câu quá ngắn sẽ được gộp lại (dưới 30 từ) để giữ ngữ điệu đọc tự nhiên và giảm số lượng phân đoạn xử lý.
* **Hiệu quả:** Đảm bảo XTTSv2 chỉ phải nhận các chuỗi văn bản cực ngắn, khống chế lượng VRAM tối đa khi inference ở mức cực kỳ thấp (~3-4 GB).

### 3.2. Giới Hạn Hàng Đợi Đơn Luồng Xử Lý GPU (Single-Worker Queue)
* **File nguồn:** [main.py](file:///f:/programfiles/AIVoice/main.py)
* **Chi tiết:** Trong khối mã điều khiển `process_single_file`, hệ thống sử dụng một `ThreadPoolExecutor` để xử lý các phân đoạn câu song song nhằm tăng hiệu năng. Tuy nhiên, đối với động cơ `clone` (XTTSv2 vốn rất nặng), hệ thống tự động thiết lập số luồng tối đa `max_workers = 1`. 
* **Hiệu quả:** Các phân đoạn câu được truyền tuần tự vào GPU để thực hiện sinh giọng nói. Điều này ngăn chặn việc nhiều luồng gửi yêu cầu inference đồng thời lên GPU gây xung đột bộ nhớ và sập card đồ họa do tràn VRAM. Đối với Piper (nhẹ) và Edge (online), số luồng được đẩy lên tương ứng là `6` và `3` luồng để tận dụng tối đa CPU và băng thông mạng.

### 3.3. Kích Hoạt PyTorch SDPA (Scaled Dot-Product Attention) Cục Bộ
* **File nguồn:** [clone.py](file:///f:/programfiles/AIVoice/src/engines/clone.py)
* **Chi tiết:** Thay vì sử dụng gói thư viện `flash-attn` (vốn cực kỳ khó cài đặt và biên dịch trên Windows), động cơ `CloneEngine` sử dụng trình quản lý ngữ cảnh (context manager) tích hợp sẵn của PyTorch 2.x:
  ```python
  sdpa_context = torch.backends.cuda.sdp_kernel(
      enable_flash=True, 
      enable_math=True, 
      enable_mem_efficient=True
  )
  ```
* **Hiệu quả:** Ép mô hình XTTSv2 sử dụng thuật toán Attention tối ưu hóa bộ nhớ và tốc độ một cách bản xứ trên GPU NVIDIA, mang lại hiệu năng tiệm cận FlashAttention nhưng hoàn toàn tương thích và không cần cài đặt dependencies phức tạp.

### 3.4. Bộ Nhớ Đệm Thông Số Giọng Mẫu (Speaker Latents Caching)
* **File nguồn:** [clone.py](file:///f:/programfiles/AIVoice/src/engines/clone.py)
* **Chi tiết:** Việc tính toán latents (`gpt_cond_latent` và `speaker_embedding`) từ file âm thanh mẫu (`ref_voice.wav`) của XTTSv2 chiếm một lượng thời gian đáng kể. Thay vì chạy lại quá trình trích xuất này cho từng câu văn trong danh sách chunk, `CloneEngine` lưu trữ kết quả này vào bộ nhớ RAM/VRAM và so sánh đường dẫn tuyệt đối của file giọng mẫu. Nếu file mẫu không đổi, nó bỏ qua bước trích xuất và nạp trực tiếp latents đã lưu vào hàm `inference`.
* **Hiệu quả:** Tiết kiệm hàng chục giây cho mỗi lượt sinh giọng nói và giảm đáng kể thao tác ghi/đọc bộ nhớ trên GPU.

### 3.5. Cắt Nhỏ Âm Thanh Khi Biến Đổi Giọng Nói (RVC Audio Slicing)
* **File nguồn:** [audio.py](file:///f:/programfiles/AIVoice/src/utils/audio.py) và [main.py](file:///f:/programfiles/AIVoice/main.py)
* **Chi tiết:** RVC (Retrieval-based Voice Conversion) thực hiện xử lý chuyển đổi giọng nói ở mức độ mẫu sóng âm thanh. Nếu xử lý một tệp âm thanh WAV dài (ví dụ: vài phút đến hàng giờ), RVC sẽ yêu cầu một lượng VRAM khổng lồ để tính toán đặc trưng và biến đổi tần số. Để giải quyết, hàm `chunk_audio_file` trong `audio.py` sẽ cắt file âm thanh đầu vào thành các đoạn WAV tạm thời dài tối đa `60 giây`. Động cơ RVC sẽ xử lý từng phân đoạn này tuần tự và ghép nối lại với nhau bằng `concatenate_wavs`.
* **Hiệu quả:** Giới hạn lượng VRAM tiêu thụ của RVC ở mức cố định, cho phép biến đổi các tệp âm thanh có thời lượng vô hạn mà không bao giờ gặp lỗi tràn VRAM.

### 3.6. Thu Hồi VRAM & Garbage Collection Triệt Để
* **File nguồn:** [rvc_engine.py](file:///f:/programfiles/AIVoice/src/engines/rvc_engine.py) và [local_ai_spice.py](file:///f:/programfiles/AIVoice/src/utils/local_ai_spice.py)
* **Chi tiết:** Cả RVC và Llama-cpp đều là các mô hình nặng. Hệ thống đảm bảo giải phóng toàn bộ tài nguyên ngay khi kết thúc tác vụ bằng khối lệnh `finally`:
  ```python
  finally:
      if rvc_instance is not None:
          try: rvc_instance.unload_model()
          except: pass
          del rvc_instance
      import gc
      gc.collect()
      if torch.cuda.is_available():
          torch.cuda.empty_cache()
  ```
* **Hiệu quả:** Trả lại VRAM sạch sẽ cho hệ điều hành ngay lập tức, ngăn ngừa hiện tượng rò rỉ bộ nhớ (memory leak) giữa các lượt chạy.

---

## 4. Các Bản Vá Tương Thích Trên Môi Trường Windows (Windows Compatibility Patches)

Để chạy trơn tru trên Windows mà không phụ thuộc vào các công cụ Linux hay các chương trình dịch biên phức tạp:

1. **Monkeypatch Torchaudio:** Trong [clone.py](file:///f:/programfiles/AIVoice/src/engines/clone.py#L30-L46), `torchaudio.load` và `torchaudio.save` được định nghĩa lại để sử dụng thư viện `soundfile` thay thế. Kỹ thuật này giúp dự án hoạt động độc lập mà **không cần cài đặt phần mềm FFmpeg** trên hệ thống Windows của người dùng.
2. **Khắc Phục Thư Viện Chuẩn Hóa Phiên Âm (vinorm):** Trong [phoneme.py](file:///f:/programfiles/AIVoice/src/utils/phoneme.py#L12-L18), hàm chuẩn hóa văn bản của thư viện `vinorm` được viết đè (`vinorm.TTSnorm = lambda t, *a, **kw: t`) để bỏ qua việc gọi một file thực thi nhị phân biên dịch cho Linux vốn sẽ gây lỗi sập trên Windows.
3. **Nạp Tự Động DLL CUDA cho ONNX Runtime-GPU:** Trong [main.py](file:///f:/programfiles/AIVoice/main.py#L9-L20), khi phát hiện hệ điều hành Windows, hệ thống tự động tìm thư mục `lib` của thư viện `torch` (nơi chứa các file CUDA DLLs đi kèm) và bổ sung nó vào đường dẫn tìm kiếm DLL (`os.add_dll_directory`). Nhờ vậy, động cơ Piper ONNX Runtime-GPU có thể chạy gia tốc phần cứng trên GPU NVIDIA mà **không yêu cầu người dùng phải tự cài đặt CUDA Toolkit và cuDNN** trên Windows.
4. **Bỏ Qua weights_only Của PyTorch 2.6+:** Trong [clone.py](file:///f:/programfiles/AIVoice/src/engines/clone.py#L75-L82), khi tải mô hình cũ qua thư viện Coqui, hệ thống tạm thời tắt cơ chế kiểm tra an toàn `weights_only` mới của PyTorch 2.6 để tránh gây ra lỗi không tương thích.

---

## 5. Hướng Dẫn Cài Đặt & Vận Hành Cho Máy Mới (Standard installation)

Môi trường chạy sử dụng thư viện cài qua pip tiêu chuẩn. Các thư viện biên dịch C++ gốc (native extension) như `coqui-tts`, `llama-cpp-python` và `fairseq` sẽ được xây dựng ổn định tại chỗ thông qua trình biên dịch MSVC chính thức của Windows.

### 5.1. Quy Trình Cài Đặt Môi Trường Chi Tiết (4 Bước)

#### **Bước 1: Cài đặt Python 3.11 (Tự động hoặc Thủ công)**
* **Tự động:** Script **[setup.bat](file:///f:/programfiles/AIVoice/setup.bat)** sẽ tự động tải và cài đặt Python 3.11.9 nếu máy chưa có.
* **Thủ công:** Tải Python 3.11 từ python.org và cài đặt (tích chọn "Add Python to PATH").

#### **Bước 2: Cài đặt Microsoft C++ Build Tools (Bắt buộc)**
* **Đường dẫn tải:** Truy cập [Microsoft Downloads](https://visualstudio.microsoft.com/downloads/) → mục "Tools for Visual Studio" → "Build Tools for Visual Studio 2022".
* **Cách chọn Workload:** Khởi chạy installer, chọn **"Desktop development with C++"** rồi nhấn Install.
* **Lưu ý:** Chỉ cần thực hiện 1 lần duy nhất trên máy mới, cài xong cần khởi động lại máy.

#### **Bước 3: Cài đặt Git for Windows**
* Hỗ trợ cài các thư viện ngữ âm trực tiếp từ Github. Tải tại [Git for Windows](https://git-scm.com/download/win).

#### **Bước 4: Chạy file setup.bat để thiết lập tự động**
* Bấm đúp file **[setup.bat](file:///f:/programfiles/AIVoice/setup.bat)**. Script tự động tạo môi trường ảo `.venv`, cấu hình pip 24.0, cài đặt dependencies trong `requirements.txt`, tải models và chạy diagnostics.


### 5.2. Chạy Kiểm Thử Tự Động
Chạy thử kịch bản kiểm thử toàn diện để xác nhận hệ thống hoạt động tốt:
```powershell
python tests/run_tests.py
```
Bộ kiểm thử sẽ kiểm tra độc lập tất cả 11 kịch bản và báo cáo kết quả.
