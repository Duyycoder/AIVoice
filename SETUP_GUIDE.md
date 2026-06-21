# 🛠️ HƯỚNG DẪN THIẾT LẬP AIVOICE TRÊN MÁY MỚI HOÀN TOÀN

Tài liệu này hướng dẫn chi tiết từng bước để cài đặt và chạy dự án **AIVoice** trên một máy tính Windows mới build (chưa cài đặt Git, Python hay bất kỳ thư viện hỗ trợ nào).

---

## BƯỚC 1: CÀI ĐẶT THỦ CÔNG CÁC PHẦN MỀM BỔ TRỢ HỆ THỐNG

Trước khi sao chép code, bạn cần tải và cài đặt thủ công các thành phần sau:

### 1. Cài đặt Driver Card Đồ Họa NVIDIA (Nếu máy có GPU)
* Để chạy mô hình nhân bản giọng nói XTTSv2 và đổi giọng RVC bằng GPU (giúp tăng tốc độ sinh giọng gấp 10-20 lần so với CPU), bạn cần cài đặt driver card màn hình mới nhất.
* **Đường dẫn tải:** [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
* **Cách thực hiện:** Chọn dòng card đồ họa tương ứng của máy mới (ví dụ GeForce RTX 5060), tải driver loại **Game Ready Driver** hoặc **Studio Driver**, cài đặt với tùy chọn mặc định và khởi động lại máy.

### 2. Cài đặt Python (Phiên bản 3.10 hoặc 3.11)
* Mô hình XTTSv2 yêu cầu phiên bản Python ổn định nhất là **3.10** hoặc **3.11** (khuyên dùng Python 3.10.11 hoặc 3.11.9). Tránh dùng Python 3.12 hoặc 3.13 vì nhiều thư viện AI chưa hỗ trợ đầy đủ.
* **Đường dẫn tải:** [Python 3.11.9 Downloads](https://www.python.org/downloads/release/python-3119/) (Cuộn xuống dưới cùng chọn bản cài đặt Windows Installer 64-bit).
* **⚠️ LƯU Ý CỰC KỲ QUAN TRỌNG:** Khi trình cài đặt Python hiện lên, bắt buộc phải tích chọn ô **"Add Python to PATH"** ở góc dưới cùng trước khi nhấn **Install Now**.

### 3. Cài đặt Microsoft C++ Build Tools (Để biên dịch thư viện AI)
* Thư viện nhân bản giọng nói `coqui-tts` chứa một số mô-đun yêu cầu biên dịch mã nguồn C++ ngay trên Windows khi cài đặt (ví dụ: tokenizer và mecab). Nếu thiếu thành phần này, quá trình cài đặt sẽ bị lỗi đỏ (lỗi thiếu compiler).
* **Đường dẫn tải:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
* **Cách thực hiện:** 
  1. Tải file cài đặt về và khởi chạy.
  2. Tại giao diện chọn gói cài đặt, tích chọn ô duy nhất: **"Desktop development with C++"** (Phát triển ứng dụng Desktop với C++).
  3. Nhấn **Install** ở góc dưới bên phải và đợi quá trình tải/cài đặt hoàn tất (khoảng 3-5 GB). Khởi động lại máy tính sau khi cài xong.

### 4. Cài đặt Git (Khuyên dùng)
* Dù không bắt buộc để chạy code, bạn vẫn cần Git để cài đặt trực tiếp các gói thư viện ngữ âm tiếng Việt phục vụ cho mô hình nhân bản giọng nói phát âm chuẩn hơn.
* **Đường dẫn tải:** [Git for Windows](https://git-scm.com/download/win) (Tải bản cài đặt 64-bit Standard). Cài đặt bằng cách bấm Next liên tục với các tùy chọn mặc định.

---

## BƯỚC 2: SAO CHÉP MÃ NGUỒN SANG MÁY MỚI

Bạn có hai cách để chuyển code sang máy mới:

1. **Cách 1 (Nhanh & Offline):** 
   * Trên máy cũ, nén thư mục dự án `AIVoice` lại thành tệp `.zip`.
   * **Lưu ý:** Hãy bỏ qua thư mục `.venv` (môi trường ảo cũ không thể chạy trên máy mới) và thư mục `models/` nếu dung lượng quá nặng.
   * Chép file `.zip` qua USB hoặc mạng nội bộ sang máy mới và giải nén vào một thư mục làm việc (ví dụ `D:\AIVoice` hoặc `C:\AIVoice`).
2. **Cách 2 (Sử dụng Git):**
   * Mở PowerShell tại thư mục làm việc trên máy mới và chạy lệnh clone từ kho lưu trữ của bạn:
     ```powershell
     git clone <đường-dẫn-repo-của-bạn>
     ```

---

## BƯỚC 3: KHỞI TẠO MÔI TRƯỜNG ẢO VÀ CÀI ĐẶT THƯ VIỆN

Mở **PowerShell** (hoặc Terminal) trên máy mới, điều hướng vào thư mục dự án vừa giải nén (ví dụ `cd D:\AIVoice`) và chạy chuỗi lệnh sau:

```powershell
# 1. Khởi tạo môi trường ảo Python cô lập
python -m venv .venv

# 2. Kích hoạt môi trường ảo vừa tạo
.venv\Scripts\Activate.ps1

# 3. Nâng cấp bộ cài đặt thư viện pip lên bản mới nhất
python -m pip install --upgrade pip

# 4. Cài đặt tất cả các thư viện lõi của dự án
pip install -r requirements.txt

# 5. Cài đặt các thư viện hỗ trợ ngữ âm tiếng Việt (Đọc chuẩn IPA)
# (Lệnh này yêu cầu hệ thống đã cài đặt Git ở Bước 1)
pip install git+https://github.com/vunb/viphoneme.git
pip install git+https://github.com/vunb/vinorm.git
```

---

## BƯỚC 4: TẢI HOẶC SAO CHÉP TRỌNG SỐ MÔ HÌNH (MODEL WEIGHTS)

Hệ thống hoạt động theo cơ chế **Offline-First**, do đó bạn cần đặt đúng file mô hình AI vào thư mục `models/`:

### 1. Tải tự động các mô hình Piper và XTTSv2
Chạy lệnh sau để hệ thống tự động tải trọng số mô hình từ Hugging Face về đúng thư mục:
```powershell
# Tải tất cả mô hình Piper và XTTSv2
python download_models.py --engine all
```

### 2. Tải/Sao chép thủ công mô hình LLM và RVC (Nếu dùng tính năng nâng cao)
Nếu bạn sử dụng tính năng viết lại văn bản bằng trí tuệ nhân tạo (Spice text) và đổi giọng Voice-to-Voice (RVC), hãy sao chép các tệp mô hình từ máy cũ sang máy mới vào đúng các đường dẫn sau:
* Mô hình LLM cục bộ: Đặt tệp `.gguf` (ví dụ `qwen2.5-1.5b-instruct-q4_k_m.gguf`) vào thư mục `models/llm/`.
* Mô hình RVC: Đặt tệp giọng nói `.pth` và file chỉ mục `.index` vào thư mục `models/rvc/`.

*Lưu ý: Nếu không sao chép từ máy cũ, bạn phải tự tải các tệp này từ trên mạng về và tạo các thư mục con `models/llm/` và `models/rvc/` nếu chúng chưa tồn tại.*

---

## BƯỚC 5: CHẨN ĐOÁN HỆ THỐNG VÀ CHẠY THỬ

### 1. Chẩn đoán phần cứng và CUDA
Hãy chạy file kiểm tra GPU để chắc chắn hệ thống mới đã nhận diện card màn hình NVIDIA và cấu trúc tăng tốc phần cứng SDPA hoạt động tốt:
```powershell
python check_gpu.py
```
* **Kết quả mong đợi:** Dòng chữ cuối cùng hiển thị `✅ GPU is ready for AIVoice inference!` và các thông số VRAM hiển thị đầy đủ.

### 2. Chạy bộ kiểm thử tự động
Để xác nhận mọi linh hồn của hệ thống (Edge online, Piper offline, XTTSv2 cloning, RVC đổi giọng) hoạt động tốt trên hệ điều hành mới, chạy bộ test tự động:
```powershell
python tests/run_tests.py
```
* **Kết quả mong đợi:** Tất cả 10 kịch bản test kết thúc với trạng thái `PASSED` và không có lỗi hệ thống phát sinh.

Bây giờ dự án của bạn đã sẵn sàng chạy ổn định trên máy mới! Gõ `python main.py` để mở giao diện Wizard tương tác tiếng Việt và bắt đầu trải nghiệm.
