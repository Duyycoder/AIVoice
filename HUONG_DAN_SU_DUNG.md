# 📖 HƯỚNG DẪN SỬ DỤNG GIAO DIỆN AIVOICE

Chào mừng bạn đến với **AIVoice** — Hệ thống chuyển đổi văn bản thành giọng nói (TTS) cục bộ, đa động cơ, hỗ trợ sao chép giọng nói (Voice Cloning) và biến đổi giọng nói (RVC Voice Conversion). 

Tài liệu này được biên soạn để giúp bạn làm chủ giao diện đồ họa (Web UI), cấu hình các thông số giọng nói chính xác và khắc phục nhanh chóng các vấn đề liên quan đến hai tính năng nâng cao đang dễ gặp lỗi là **RVC** và **viphoneme (phomie)**.

---

## 🚀 1. Hướng Dẫn Khởi Chạy Giao Diện (Web UI)

Để khởi chạy giao diện Web UI cục bộ:
1. **Cách nhanh nhất:** Bấm đúp vào file script **`chay_giao_dien.bat`** ở thư mục gốc của dự án (bạn có thể tạo Shortcut cho file này ra ngoài màn hình Desktop để mở nhanh).
2. **Khởi chạy bằng dòng lệnh:** Mở Command Prompt hoặc PowerShell tại thư mục dự án và chạy:
   ```powershell
   python web_ui.py
   ```
3. Sau khi khởi chạy thành công, trình duyệt của bạn sẽ tự động mở trang web (hoặc bạn có thể truy cập thủ công qua địa chỉ): **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## 🎛️ 2. Hướng Dẫn Thiết Lập Các Option Giọng Nói (Voice Options)

Giao diện Web UI của AIVoice chia thành các phần cấu hình trực quan như sau:

### A. Chọn Động Cơ Chuyển Đổi (TTS Engine)
Tại ô **Chọn Động Cơ**, bạn có thể lựa chọn 3 công nghệ khác nhau:
1. **Edge-TTS (Online Cloud):**
   * *Đặc điểm:* Chạy trực tuyến (yêu cầu kết nối Internet), tốc độ xử lý rất nhanh, giọng đọc truyền cảm tự nhiên và hoàn toàn miễn phí.
   * *Thiết lập:* 
     * Chọn giọng đọc trong ô **Giọng đọc Edge-TTS** (ví dụ: `vi-VN-NamMinhNeural` cho giọng Nam miền Bắc, `vi-VN-HoaiMyNeural` cho giọng Nữ miền Nam).
     * Điều chỉnh **Tốc độ giọng đọc (Speed)** từ `0.5` đến `2.0` (mặc định là `1.0`).
2. **Piper TTS (Offline ONNX):**
   * *Đặc điểm:* Hoạt động hoàn toàn ngoại tuyến (Offline), siêu nhẹ, chạy mượt mà ngay cả trên máy tính không có card đồ họa (chỉ cần CPU).
   * *Thiết lập:* 
     * Chọn tệp mô hình trong ô **Model ONNX** (ví dụ: `vi_VN-vais1000-medium.onnx`). Các tệp này phải được tải về và đặt trong thư mục `models/piper/`.
     * Nhập **ID Người nói (Speaker ID)** (để mặc định là `0`).
3. **Clone TTS (Offline XTTSv2):**
   * *Đặc điểm:* Sao chép giọng nói của bất kỳ ai từ một tệp âm thanh mẫu ngắn. Hoạt động ngoại tuyến, yêu cầu máy tính có card đồ họa NVIDIA (khuyên dùng VRAM từ 6GB trở lên như RTX 3060).
   * *Thiết lập:*
     * Chọn tệp âm thanh giọng mẫu tại ô **Giọng nói mẫu (Reference Audio)**. Tệp mẫu là file `.wav` sạch (không lẫn tạp âm, nhạc nền), dài từ 5 - 10 giây, được đặt trong thư mục `data/voices/`.
     * Chọn **Ngôn ngữ (Language Code)**: `vi` (Tiếng Việt) hoặc `en` (Tiếng Anh).

### B. Tùy Chọn Nâng Cao & Hậu Xử Lý (Post-Processing)
* **Chuẩn hóa LUFS (`--normalize`):** Bật tính năng này giúp âm lượng của tệp đầu ra đạt tiêu chuẩn phát thanh chuyên nghiệp (mặc định là `-14.0 LUFS`), giúp âm thanh đồng đều, không bị quá to hoặc quá nhỏ.
* **Khoảng lặng giữa các câu (s):** Thời gian nghỉ (giây) giữa các câu khi ghép nối (mặc định là `0.3`). Bạn có thể tăng lên `0.5` hoặc `0.8` nếu muốn giọng đọc chậm rãi, thong thả hơn.
* **Thời gian Fade In / Fade Out (s):** Tạo hiệu ứng lớn dần ở đầu file và nhỏ dần ở cuối file (mặc định `0.1`), giúp triệt tiêu các tiếng "bụp" nhiễu âm thanh khi bắt đầu/kết thúc phát.

### C. Viết Lại Cảm Xúc Bằng LLM Cục Bộ (`--spice_text`)
* Khi tích chọn, hệ thống sẽ gửi văn bản thô qua mô hình ngôn ngữ lớn chạy cục bộ (ví dụ: `qwen2.5-1.5b-instruct-q4_k_m.gguf` đặt tại `models/llm/`) để viết lại văn bản theo phong cách bạn chọn (Tếu táo, hài hước, thân thiện...) trước khi đưa vào sinh giọng nói.

---

## 🛠️ 3. Phân Tích & Khắc Phục Lỗi Hai Tính Năng: RVC và viphoneme (phomie)

Đây là hai tính năng nâng cao giúp nâng tầm chất lượng giọng đọc nhưng rất dễ xảy ra lỗi trên hệ điều hành Windows do xung đột thư viện hoặc tính tương thích phần cứng.

---

### 🔴 Tính Năng 1: Đổi Giọng Bằng Mô Hình RVC (Retrieval-based Voice Conversion)

RVC là công nghệ Voice-to-Voice (Biến đổi âm thanh đầu ra của TTS thành giọng của một nhân vật cụ thể bằng AI). 

#### ⚠️ Các lỗi thường gặp và cách khắc phục:

1. **Lỗi tràn bộ nhớ GPU (VRAM Out Of Memory - OOM):**
   * *Triệu chứng:* Đang xử lý đổi giọng thì cửa sổ terminal bị tắt đột ngột, hoặc báo lỗi `CUDA out of memory`. RVC hoạt động ở mức độ sóng âm (waveform) nên ngốn cực kỳ nhiều VRAM khi xử lý file dài.
   * *Giải pháp:*
     * **Cơ chế tự động chia đoạn (đã tích hợp):** Hệ thống AIVoice tự động cắt file âm thanh ra thành các phân đoạn nhỏ tối đa `60 giây` để xử lý RVC tuần tự, sau đó ghép lại. Điều này giúp kiểm soát lượng VRAM tiêu thụ ở mức cố định.
     * **Chọn Hardware Profile phù hợp:** Trên Web UI, ở mục tùy chọn thiết bị, hãy chọn đúng cấu hình:
       * Chọn **RTX 3060** nếu card của bạn có 6GB VRAM.
       * Chọn **CPU Mode** nếu bạn muốn chạy an toàn trên RAM máy tính (chấp nhận tốc độ chậm hơn).
       * Không chọn **RTX 5060** nếu card đồ họa thực tế của bạn có dung lượng VRAM dưới 8GB.

2. **Lỗi "CUDA error: no kernel image is available for execution on the device":**
   * *Triệu chứng:* Lỗi xảy ra trên các máy tính sử dụng card đồ họa dòng **RTX 50-Series (RTX 5060, 5070...)** chạy kiến trúc Blackwell mới. Các phiên bản PyTorch cài mặc định không chứa mã chạy (compiled kernel) cho kiến trúc mới này.
   * *Giải pháp:*
     * Gỡ cài đặt PyTorch hiện tại và cài đặt phiên bản hỗ trợ CUDA 12.8 bằng cách mở terminal và chạy lệnh:
       ```powershell
       .\.venv\Scripts\pip uninstall torch torchvision torchaudio -y
       .\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
       ```
     * Chạy kiểm tra lại bằng nút **Chẩn đoán GPU** trên giao diện Web (hoặc chạy lệnh `python check_gpu.py`).

3. **Lỗi xung đột thư viện Numpy hoặc lỗi cài đặt gói `rvc-python`:**
   * *Triệu chứng:* Thư viện `rvc-python` yêu cầu `numpy<=1.23.5`, trong khi động cơ Clone (`coqui-tts`) bắt buộc phải dùng `numpy>=1.26.0`. Việc cài đặt thông thường qua `pip install -r requirements.txt` sẽ dẫn đến xung đột phiên bản và làm hỏng một trong hai tính năng.
   * *Giải pháp:*
     * Bắt buộc phải cài đặt thông qua file **`setup.bat`**. Script này đã được tối ưu hóa để cài đặt `rvc-python` với tùy chọn `--no-deps` (không cài đặt các thư viện phụ thuộc đi kèm để tránh ghi đè numpy) và cài đặt thủ công các gói bổ trợ tương thích ở phiên bản phù hợp.
     * Đảm bảo máy tính đã cài đặt **Microsoft C++ Build Tools** (Workload: *Desktop development with C++*) trước khi chạy cài đặt, vì RVC yêu cầu biên dịch mã nguồn C++ cho các gói `fairseq` và `pyworld`.

4. **Không thấy mô hình RVC hiển thị trên Web UI:**
   * *Giải pháp:* Các tệp mô hình RVC phải được đặt vào đúng thư mục:
     * Tệp mô hình chính dạng `.pth` đặt vào: `models/rvc/`
     * Tệp chỉ mục hỗ trợ dạng `.index` đặt vào: `models/rvc/` (Tùy chọn, giúp nâng cao độ giống giọng).

---

### 🟢 Tính Năng 2: Phiên Âm IPA Tiếng Việt (viphoneme / phomie)

Khi sử dụng tính năng Clone (sao chép giọng mẫu XTTSv2), mô hình AI gốc được huấn luyện chủ yếu bằng tiếng Anh/đa ngôn ngữ chuẩn quốc tế. Khi đọc tiếng Việt trực tiếp, mô hình rất dễ phát âm sai, nói lắp hoặc nuốt chữ. 
Tính năng **`--phonemize`** sử dụng thư viện `viphoneme` để dịch chữ tiếng Việt thành ký tự phiên âm quốc tế IPA trước khi chuyển cho mô hình AI, giúp giọng đọc rõ chữ và tự nhiên hơn rất nhiều.

#### ⚠️ Các lỗi thường gặp và cách khắc phục:

1. **Lỗi crash chương trình trên hệ điều hành Windows do thư viện `vinorm`:**
   * *Triệu chứng:* Lỗi phát sinh từ thư viện chuẩn hóa văn bản `vinorm` đi kèm với `viphoneme`. Thư viện này chứa một file thực thi C nhị phân được biên dịch trên Linux, nên khi chạy trên Windows sẽ lập tức gây crash tiến trình.
   * *Giải pháp:*
     * **Cơ chế Monkey-patch (đã tích hợp):** File mã nguồn `src/utils/phoneme.py` của hệ thống đã tự động ghi đè (`monkey-patch`) hàm `vinorm.TTSnorm` để bỏ qua phần chạy file nhị phân lỗi trên Windows, trả về văn bản gốc an toàn. 
     * Bạn chỉ cần kích hoạt tính năng **Phiên âm IPA tiếng Việt (--phonemize)** bằng nút gạt trên Web UI mà không cần lo lắng về lỗi crash này.

2. **Gạt nút bật Phiên âm IPA nhưng giọng đọc không thay đổi (Lỗi âm thầm bỏ qua):**
   * *Triệu chứng:* Hệ thống không báo lỗi lớn nhưng trong cửa sổ nhật ký (Console Log) xuất hiện dòng cảnh báo: `Warning: Phonemization failed with error... Falling back to raw text.` và giọng đọc vẫn bị nói lắp/nuốt chữ như cũ.
   * *Nguyên nhân:* Môi trường ảo `.venv` chưa được cài đặt thư viện `viphoneme` và `vinorm` (thường do máy thiếu công cụ Git lúc chạy `setup.bat`).
   * *Giải pháp:*
     1. Tải và cài đặt **[Git for Windows](https://git-scm.com/download/win)**.
     2. Mở cửa sổ terminal tại thư mục dự án và chạy lệnh sau để cài đặt trực tiếp từ GitHub vào môi trường ảo:
        ```powershell
        .venv\Scripts\pip install git+https://github.com/vunb/viphoneme.git git+https://github.com/vunb/vinorm.git
        ```
     3. Khởi chạy lại giao diện Web UI và thử lại. Dấu hiệu nhận biết tính năng hoạt động là trong Console Log sẽ in ra các chuỗi ký tự IPA dạng ngữ âm thay vì chữ viết thông thường trước khi mô hình Clone xử lý.

---

## 💡 4. Mẹo Sử Dụng Giao Diện Hiệu Quả (Tips & Tricks)

* **Sử dụng Serialization Lock (Hàng đợi an toàn):** Hệ thống tích hợp sẵn khóa luồng xử lý trên GPU. Nếu bạn nhấn nút sinh âm thanh nhiều lần hoặc có nhiều tab trình duyệt gửi yêu cầu cùng lúc, hệ thống sẽ xếp các yêu cầu sau vào hàng đợi `[Xếp hàng chờ xử lý (GPU đang bận)]` thay vì chạy song song gây sập GPU (OOM). Đừng lo lắng nếu thấy tiến trình ở trạng thái chờ, hãy kiên nhẫn đợi file xử lý xong.
* **Xử lý tệp ngoài thư mục dự án:** Web UI của AIVoice cho phép bạn nhập trực tiếp đường dẫn tuyệt đối của tệp tin ở ổ đĩa khác (ví dụ: `D:\Audiobook\chuong_1.md`) và thư mục xuất âm thanh (ví dụ: `E:\ThanhPham`). Điều này giúp bạn tiết kiệm dung lượng ổ cứng hệ thống (ổ C) khi xử lý các dự án truyện dài tập.
* **Thường xuyên sử dụng GPU Diagnostic (Chẩn đoán GPU):** Khi gặp bất kỳ sự cố nào liên quan đến tốc độ sinh giọng đọc (bị chậm) hoặc lỗi CUDA, hãy cuộn xuống cuối giao diện Web UI và nhấn nút **Chẩn đoán GPU**. Báo cáo chi tiết về tình trạng card đồ họa, dung lượng VRAM còn trống và tính khả dụng của nhân tính toán Tensor Cores (SDPA) sẽ giúp bạn nhanh chóng khoanh vùng nguyên nhân lỗi.
