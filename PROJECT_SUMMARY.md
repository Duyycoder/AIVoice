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
│   └── run_tests.py       # Tập lệnh chạy tự động 10 kịch bản kiểm thử toàn diện
│
├── check_gpu.py           # Công cụ chẩn đoán khả năng tương thích GPU/CUDA và kiểm tra tính năng SDPA
├── download_models.py     # Tập lệnh tự động tải xuống các mô hình (Piper & XTTSv2) từ Hugging Face
├── main.py                # Điểm khởi chạy (Entry Point) của CLI và Wizard tương tác từng bước
├── requirements.txt       # Danh sách thư viện và dependencies
└── plan.md                # Kế hoạch phát triển dự án và trạng thái hiện tại
```

---

## 3. Các Giải Pháp Tối Ưu Hóa Tránh Tràn VRAM (OOM) Cho GPU 8GB (RTX 5060)

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

## 5. Chi Tiết Các Tệp Tin Mã Nguồn Chính

### 5.1. Điểm Vào Điều Khiển - `main.py`
Chứa các thành phần chính:
* **Interactive Wizard Mode:** Nếu chạy không có tham số, nó sẽ quét các thư mục `data/inputs/`, `data/voices/` và `models/` để hiển thị menu lựa chọn thân thiện cho người dùng, sau đó sinh lệnh CLI và thực thi.
* **CLI Parser:** Định nghĩa tất cả các tham số từ tốc độ, hiệu ứng âm thanh đến đường dẫn mô hình.
* **Pipeline Orchestrator:** Liên kết các bước làm sạch văn bản -> phân đoạn câu -> sinh âm thanh từng phần -> ghép nối -> chuẩn hóa âm lượng LUFS -> hậu xử lý fade -> áp dụng RVC.

### 5.2. Động Cơ Nhân Bản - `src/engines/clone.py`
Tích hợp Coqui XTTSv2 cục bộ:
* Chịu trách nhiệm khởi tạo lớp `TTS` từ đường dẫn cục bộ chỉ định (tránh kết nối internet).
* Thực hiện patch tokenizer để hỗ trợ việc phân mảnh các từ tiếng Việt (`VoiceBpeTokenizer.preprocess_text` được sửa đổi để hỗ trợ mã ngôn ngữ `vi`).
* Thực hiện sinh giọng nói và lưu kết quả ra tần số lấy mẫu chuẩn của XTTSv2 (24kHz).

### 5.3. Động Cơ Chuyển Đổi Giọng RVC - `src/engines/rvc_engine.py`
Tích hợp thư viện `rvc-python`:
* Hỗ trợ hai phương thức: `apply_rvc` cho một tệp đơn lẻ và `apply_rvc_to_segments` cho danh sách phân đoạn câu (giúp tối ưu hóa VRAM).
* Sử dụng bộ dự đoán cao độ `rmvpe` chất lượng cao và hỗ trợ tham số dịch giọng (`pitch_shift`).

### 5.4. Tiện Ích Hậu Xử Lý Âm Thanh - `src/utils/audio.py`
* **`concatenate_wavs`:** Ghép nối các file WAV thô của các đoạn câu, tự động chèn khoảng lặng tĩnh (`silence_duration`) giữa các câu và thực hiện Peak Normalization lên mức 0.95 để tránh méo tiếng.
* **`apply_audio_post_processing`:** Áp dụng thuật toán fade-in (tăng âm lượng đầu file từ 0.0 lên 1.0) và fade-out (giảm âm lượng cuối file về 0.0) tuyến tính. Đo lường và chuẩn hóa âm lượng tích hợp theo tiêu chuẩn phát thanh chuyên nghiệp **-14 LUFS** bằng thư viện `pyloudnorm`.

---

## 6. Hướng Dẫn Vận Hành & Chạy Thử (Dành Cho Claude Code)

### 6.1. Thiết Lập Môi Trường
Mở PowerShell trong thư mục gốc của dự án:
```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\Activate.ps1

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt

# Cài đặt thêm các thư viện ngữ âm tiếng Việt (tùy chọn nhưng khuyên dùng)
pip install git+https://github.com/vunb/viphoneme.git
pip install git+https://github.com/vunb/vinorm.git
```

### 6.2. Kiểm Trạng Thái GPU & CUDA
Chạy file chẩn đoán để xác định trạng thái GPU và khả năng chạy SDPA:
```powershell
python check_gpu.py
```

### 6.3. Tải Trọng Số Mô HÌnh Cục Bộ
Sử dụng script tải chuyên dụng để chuẩn bị sẵn các mô hình ngoại tuyến:
```powershell
# Tải tất cả các mô hình cần thiết (Piper & XTTSv2)
python download_models.py --engine all
```
*Lưu ý: Mô hình LLM cục bộ (file `.gguf`) và mô hình RVC (file `.pth`/`.index`) cần được người dùng đặt thủ công vào đúng cấu trúc thư mục trong `models/llm/` và `models/rvc/`.*

### 6.4. Chạy Bộ Kiểm Thử Tự Động (10 Kịch Bản)
Chạy bộ kiểm thử tự động để xác nhận toàn bộ hệ thống hoạt động ổn định:
```powershell
python tests/run_tests.py
```
Bộ kiểm thử sẽ kiểm tra độc lập các chức năng sinh giọng nói (Edge, Piper, XTTSv2 tiếng Việt, XTTSv2 tiếng Anh, Batch processing, Config override, RVC standalone, lỗi tham số đầu vào...).

---

## 7. Các Lệnh Chạy CLI Thực Tế Tham Khảo

1. **Sinh giọng nói bằng Edge-TTS (Online, nhanh):**
   ```powershell
   python main.py --input data/inputs/test_vi.md --engine edge --voice vi-VN-NamMinhNeural --speed 1.0
   ```
2. **Sinh giọng nói ngoại tuyến bằng Piper ONNX:**
   ```powershell
   python main.py --input data/inputs/test_vi.md --engine piper --model models/piper/vi_VN-vais1000-medium.onnx
   ```
3. **Sao chép giọng nói ngoại tuyến bằng XTTSv2:**
   ```powershell
   python main.py --input data/inputs/test_vi.md --engine clone --model models/xtts_v2 --ref_audio data/voices/ref_voice.wav --voice vi --phonemize
   ```
4. **Chạy trọn gói chuỗi xử lý (LLM viết lại + Edge sinh âm + RVC đổi giọng):**
   ```powershell
   python main.py --input data/inputs/test_vi.md --engine edge --voice vi-VN-NamMinhNeural --spice_text --llm_model models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf --rvc_model models/rvc/adam.pth --rvc_pitch 0
   ```
5. **Chạy đổi giọng trực tiếp cho file âm thanh bằng RVC:**
   ```powershell
   python main.py --input data/voices/ref_voice.wav --engine rvc --rvc_model models/rvc/ElevenLabs_Adam_FR.pth --rvc_index models/rvc/added_IVF4988_Flat_nprobe_1_ElevenLabs_Adam_FR_v2.index
   ```
