# Hướng Dẫn Cài Đặt GPU Cho Kiến Trúc Blackwell (RTX 50-Series / RTX 5060)

Tài liệu này hướng dẫn cách cấu hình PyTorch hỗ trợ tăng tốc GPU trên card đồ họa NVIDIA RTX 50-series (kiến trúc Blackwell, Compute Capability 12.0 / `sm_120`).

---

## 1. Vấn Đề Tương Thích
Card đồ họa RTX 5060/5070/5080/5090 sử dụng kiến trúc Blackwell rất mới với Compute Capability 12.0 (`sm_120`). 
* Các bản build PyTorch thông thường (như CUDA 12.1 hoặc CUDA 12.4 cài từ file local `.whl`) **không có sẵn compiled kernel** cho kiến trúc `sm_120`. 
* Khi chạy, PyTorch vẫn nhận dạng được tên GPU nhưng mọi phép toán CUDA thực tế sẽ bị lỗi crash lập tức:
  `CUDA error: no kernel image is available for execution on the device`

---

## 2. Cách Khắc Phục

Để chạy ổn định trên GPU RTX 5060, bạn bắt buộc phải cài đặt PyTorch thông qua chỉ mục **CUDA 12.8 (`cu128`)** hoặc bản **nightly `cu128`** vì các phiên bản này đã tích hợp đầy đủ mã nhị phân tương thích cho kiến trúc Blackwell.

### Quy Trình Cài Đặt Hợp Lệ:

1. **Gỡ cài đặt PyTorch CPU/CUDA cũ:**
   Chạy lệnh sau trong terminal để dọn dẹp sạch sẽ:
   ```powershell
   .\.venv\Scripts\pip uninstall torch torchvision torchaudio -y
   ```

2. **Cài đặt lại PyTorch qua chỉ mục CUDA 12.8:**
   ```powershell
   # Bản Stable hỗ trợ CUDA 12.8
   .\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

3. **(Tùy chọn) Bản Nightly nếu bản Stable trên chưa có sẵn:**
   Nếu bản Stable chưa hỗ trợ đầy đủ hoặc báo lỗi, hãy thử bản Pre-release (Nightly):
   ```powershell
   .\.venv\Scripts\pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

4. **Xác nhận trạng thái GPU:**
   ```powershell
   .\.venv\Scripts\python.exe check_gpu.py
   ```
   *Đảm bảo SDPA test và GPU tensor test đều báo tích xanh (✅).*

---

## 3. Lưu Ý Đặc Biệt
> [!IMPORTANT]
> **KHÔNG** sử dụng `pip install torch` mặc định hoặc cài đặt từ các file `.whl` tải thủ công cho CUDA 12.1/12.4 vì chúng sẽ thiếu kernel biên dịch sẵn cho `sm_120`. Luôn sử dụng index `cu128` hoặc bản `nightly/cu128` cho dòng RTX 50-series.
