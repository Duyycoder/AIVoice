import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import sys
if __name__ == "__main__":
    # DLL collision safety: Mock torch to prevent modelscope/paddlex from loading real PyTorch CUDA DLLs
    import types
    from unittest.mock import MagicMock

    class MockMetaclass(type):
        def __getattr__(cls, name):
            return MockClass
        def __call__(cls, *args, **kwargs):
            return MockInstance()

    class MockClass(metaclass=MockMetaclass):
        pass

    class MockInstance:
        def __getattr__(self, name):
            return self
        def __call__(self, *args, **kwargs):
            return self
        def __bool__(self):
            return False
        def __iter__(self):
            return iter([])

    KNOWN_CLASSES = {'device', 'dtype', 'Tensor', 'Module', 'Parameter'}

    class DynamicMockModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__spec__ = sys.modules['os'].__spec__
            
        def __getattr__(self, name):
            if name[0].isupper() or name in KNOWN_CLASSES:
                return MockClass
            sub_name = f"{self.__name__}.{name}"
            if sub_name not in sys.modules:
                sys.modules[sub_name] = DynamicMockModule(sub_name)
            sub_m = sys.modules[sub_name]
            setattr(self, name, sub_m)
            return sub_m

        def __call__(self, *args, **kwargs):
            return self

        def __bool__(self):
            return False

    torch_mock = DynamicMockModule('torch')
    sys.modules['torch'] = torch_mock

    submodules = [
        'torch.multiprocessing', 'torch.distributed', 'torch.nn', 'torch.utils',
        'torch.utils.data', 'torch.cuda', 'torch.jit', 'torch.optim', 'torch.autograd',
        'torch.nn.functional', 'torch.nn.init', 'torch.nn.parameter', 'torch.nn.modules',
    ]
    for sub in submodules:
        sub_mock = DynamicMockModule(sub)
        sys.modules[sub] = sub_mock
        parts = sub.split('.')
        parent = torch_mock
        for part in parts[1:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], sub_mock)

import paddle
import subprocess
import json
from app.utils import utils

def log_json(event: str, data: dict):
    """Outputs progress log as a JSON string to stdout."""
    print(json.dumps({"event": event, **data}, ensure_ascii=False))
    sys.stdout.flush()

def grab_preview_frame(video_path: str, out_image: str, at_seconds: float = None) -> dict:
    """Lấy 1 frame (mặc định ~giữa video nếu at_seconds=None) ghi ra out_image (jpg).
    Trả về {'image': out_image, 'width': W, 'height': H, 'duration': D} (kích thước gốc để UI map toạ độ)."""
    # Metadata (W/H/duration) lấy bằng moviepy — KHÔNG dùng ffprobe (CB5)
    from moviepy.video.io.VideoFileClip import VideoFileClip
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    clip = VideoFileClip(video_path)
    w, h = clip.size
    dur = clip.duration
    clip.close()
    
    t = at_seconds if at_seconds is not None else dur / 2.0
    
    ffmpeg_bin = utils.get_ffmpeg_binary()
    
    # Extract frame using ffmpeg
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{t:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "3",
        out_image
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to grab frame using ffmpeg: {res.stderr}")
        
    return {
        "image": os.path.abspath(out_image),
        "width": w,
        "height": h,
        "duration": dur
    }

def _chon_model_ocr(kich_co: str) -> None:
    """videocr tạo PaddleOCR không tên model → thư viện lấy PP-OCRv6_medium. Chọn bộ nhỏ hơn bằng cách bọc hàm tạo."""
    if kich_co not in ("small", "tiny"):
        return
    import videocr.video as vv
    goc = getattr(vv.PaddleOCR, "_goc", vv.PaddleOCR)

    def tao(*a, **kw):
        if not kw.get("text_detection_model_name"):
            kw["text_detection_model_name"] = f"PP-OCRv6_{kich_co}_det"
        if not kw.get("text_recognition_model_name"):
            kw["text_recognition_model_name"] = "PP-OCRv6_small_rec"     # không có tiny_rec
        return goc(*a, **kw)
    tao._goc = goc
    vv.PaddleOCR = tao


def _doc_khung_ffmpeg(path: str, vung: tuple, ocr_fps: float, bat_dau: float, thoi_luong: float, dung_gpu: bool):
    """Sinh (k, ảnh BGR) — ffmpeg giải mã (NVDEC nếu có), lọc còn `ocr_fps` khung/s và cắt sẵn vùng phụ đề.
    videocr dùng OpenCV giải mã MỌI khung 1080p bằng CPU (kể cả khung bỏ qua): ~170 ms/khung trong khi OCR chỉ 34 ms (đo 29/09)."""
    import numpy as np
    x, y, w, h = vung
    w, h = w - w % 2, h - h % 2
    lenh = [utils.get_ffmpeg_binary(), "-hide_banner", "-loglevel", "error"]
    if dung_gpu:
        lenh += ["-hwaccel", "cuda"]
    if bat_dau > 0:
        lenh += ["-ss", f"{bat_dau:.3f}"]
    lenh += ["-i", path]
    if thoi_luong > 0:
        lenh += ["-t", f"{thoi_luong:.3f}"]
    lenh += ["-an", "-vf", f"fps={ocr_fps:g},crop={w}:{h}:{x}:{y}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(lenh, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 4)
    co, k = w * h * 3, 0
    try:
        while True:
            buf = p.stdout.read(co)
            if len(buf) < co:
                break
            yield k, np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()   # copy: frombuffer chỉ đọc
            k += 1
    finally:
        p.stdout.close()
        loi = p.stderr.read().decode("utf-8", "replace")
        p.wait()
    if k == 0:
        raise RuntimeError(f"ffmpeg không đọc được khung nào: {loi[-300:]}")


def _va_doc_bang_ffmpeg(ocr_fps: float) -> None:
    """Thay vòng đọc khung của videocr (Video.run_ocr) bằng ffmpeg; giữ nguyên OCR, lọc khung giống nhau và gộp câu.
    Mỗi khung OCR phủ luôn các khung nằm giữa 2 lần đọc — videocr chỉ gộp 2 khung cách ≤ 0,09 s, đọc thưa mà không phủ
    thì 1 câu thành 5 câu dài 0,016 s (đo 29/09)."""
    import inspect
    import cv2
    import numpy as np
    import videocr.video as vv
    from videocr.models import PredictedFrames
    from videocr import utils as vutils
    goc = getattr(vv.Video.run_ocr, "_goc", vv.Video.run_ocr)
    chu_ky = inspect.signature(goc)

    def run_ocr(self, *a, **kw):
        t = chu_ky.bind(self, *a, **kw).arguments
        self.lang, self.use_fullframe, self.pred_frames = t["lang"], t["use_fullframe"], []
        ocr = vv.PaddleOCR(lang=self.lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False, device="gpu" if t["use_gpu"] else "cpu")
        if self.use_fullframe:
            vung = (0, 0, self.width, self.height)
        elif t["crop_width"] and t["crop_height"]:
            cx, cy = max(0, int(t["crop_x"] or 0)), max(0, int(t["crop_y"] or 0))
            vung = (cx, cy, min(int(t["crop_width"]), self.width - cx), min(int(t["crop_height"]), self.height - cy))
        else:
            vung = (0, 2 * self.height // 3, self.width, self.height - 2 * self.height // 3)   # như videocr: 1/3 dưới
        dau = vutils.get_frame_index(t["time_start"], self.fps) if t["time_start"] else 0
        cuoi = vutils.get_frame_index(t["time_end"], self.fps) if t["time_end"] else self.num_frames
        buoc = self.fps / ocr_fps                  # số khung gốc giữa 2 lần đọc
        tong = max(1, int((cuoi - dau) / buoc))
        nguong_sang, nguong_giong, nguong_diem = t["brightness_threshold"], t["similar_image_threshold"], t["similar_pixel_threshold"]
        conf = float(t["conf_threshold"]) / 100
        truoc, khung_cuoi = None, None

        def cac_khung():
            doc = lambda gpu: _doc_khung_ffmpeg(self.path, vung, ocr_fps, dau / self.fps,
                                                (cuoi - dau) / self.fps if t["time_end"] else 0, gpu)
            if not t["use_gpu"]:
                yield from doc(False)
                return
            da_co = False
            try:
                for x in doc(True):
                    da_co = True
                    yield x
            except RuntimeError as e:
                if da_co:
                    raise
                log_json("ocr_progress", {"message": f"Không giải mã GPU được ({e}) — chuyển sang CPU"})
                yield from doc(False)

        for k, anh in cac_khung():
            i = dau + int(round(k * buoc))
            phu = i + max(0, int(round(buoc)) - 1)
            if nguong_sang:
                anh = cv2.bitwise_and(anh, anh, mask=cv2.inRange(anh, (nguong_sang,) * 3, (255, 255, 255)))
            if nguong_giong:
                xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
                if truoc is not None and khung_cuoi is not None:
                    _, khac = cv2.threshold(cv2.absdiff(truoc, xam), nguong_diem, 255, cv2.THRESH_BINARY)
                    if np.count_nonzero(khac) < nguong_giong:
                        khung_cuoi.end_index = phu
                        truoc = xam
                        continue
                truoc = xam
            khung_cuoi = PredictedFrames(i, ocr.ocr(anh), conf)
            khung_cuoi.end_index = phu
            self.pred_frames.append(khung_cuoi)
            if k % 50 == 0:
                log_json("ocr_progress", {"message": f"OCR {min(k, tong)}/{tong} khung", "percent": round(min(k, tong) / tong * 100, 1)})
    run_ocr._goc = goc
    vv.Video.run_ocr = run_ocr


def extract_hardsub_ocr_srt(
    video_path: str,
    output_srt: str,
    lang: str = "ch",                # "ch" (Trung) | "en" (Anh)
    crop: tuple = None,              # (x, y, w, h)
    time_start: str = "",            # "" hoặc "mm:ss"
    time_end: str = "",
    conf_threshold: int = 75,
    sim_threshold: int = 80,
    frames_to_skip: int = None,
    use_gpu: bool = False,
    progress_cb=None,
    ocr_fps: float = 10.0,           # số khung OCR mỗi giây video (phụ đề hiện ≥ ~1 s nên 10 là đủ)
    ocr_model: str = "medium",       # PP-OCRv6: medium (mặc định thư viện) | small (nhanh hơn)
) -> str:
    """Gọi videocr-PaddleOCR: save_subtitles_to_file."""
    log_json("ocr_start", {"video_path": video_path, "output_srt": output_srt, "lang": lang})

    # Trước 29/09 videocr OCR mọi khung thứ 2 (60fps = 30 khung/s) và giải mã mọi khung bằng CPU: 1 phút video ~220 s.
    ocr_fps = max(1.0, float(ocr_fps))
    _chon_model_ocr(ocr_model)
    _va_doc_bang_ffmpeg(ocr_fps)
    log_json("ocr_progress", {"message": f"OCR {ocr_fps:g} khung/giây, model {ocr_model}, giải mã {'GPU' if use_gpu else 'CPU'}"})
    
    try:
        from videocr import save_subtitles_to_file
    except ImportError:
        log_json("ocr_error", {"error": "Thư viện videocr-PaddleOCR chưa được cài đặt."})
        raise ImportError("videocr-PaddleOCR is not installed in the virtual environment.")
        
    crop_x = None
    crop_y = None
    crop_width = None
    crop_height = None
    use_fullframe = True
    
    if crop and len(crop) == 4:
        x, y, w, h = crop
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            crop_x = x
            crop_y = y
            crop_width = w
            crop_height = h
            use_fullframe = False
            log_json("ocr_roi", {"crop_x": x, "crop_y": y, "crop_width": w, "crop_height": h})
            
    try:
        # Note: oliverfei/videocr-PaddleOCR save_subtitles_to_file parameters
        # Some versions might call it crop_width/crop_height.
        # Let's dynamically pass these parameters or try to handle errors.
        kwargs = {
            "video_path": video_path,
            "file_path": output_srt,
            "lang": lang,
            "time_start": time_start or "0:00",
            "time_end": time_end or "",
            "conf_threshold": conf_threshold,
            "sim_threshold": sim_threshold,
            "use_fullframe": use_fullframe,
            "use_gpu": use_gpu,
            "frames_to_skip": frames_to_skip
        }
        
        if not use_fullframe:
            kwargs["crop_x"] = crop_x
            kwargs["crop_y"] = crop_y
            kwargs["crop_width"] = crop_width
            kwargs["crop_height"] = crop_height
            
        if use_gpu:
            try:
                log_json("ocr_progress", {"message": "Đang chạy trích xuất phụ đề OCR (PaddleOCR) trên GPU..."})
                save_subtitles_to_file(**kwargs)
            except Exception as e:
                log_json("ocr_progress", {"message": f"Lỗi chạy GPU OCR ({e}). Đang tự động chuyển sang chế độ CPU..."})
                kwargs["use_gpu"] = False
                save_subtitles_to_file(**kwargs)
        else:
            log_json("ocr_progress", {"message": "Đang chạy trích xuất phụ đề OCR (PaddleOCR) trên CPU..."})
            save_subtitles_to_file(**kwargs)
        
        if not os.path.exists(output_srt) or os.path.getsize(output_srt) == 0:
            raise RuntimeError("OCR failed to generate subtitle file or generated file is empty.")
            
        log_json("ocr_done", {"output_srt": output_srt})
        return output_srt
    except Exception as e:
        log_json("ocr_error", {"error": str(e)})
        raise e
    finally:
        # Giải phóng VRAM paddle sau OCR
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone subtitle extractor")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-srt", required=True)
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--crop-x", type=int, default=-1)
    parser.add_argument("--crop-y", type=int, default=-1)
    parser.add_argument("--crop-w", type=int, default=-1)
    parser.add_argument("--crop-h", type=int, default=-1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--ocr-fps", type=float, default=10.0)
    parser.add_argument("--ocr-model", default="medium", choices=["medium", "small", "tiny"])
    args = parser.parse_args()

    crop = None
    if args.crop_x >= 0 and args.crop_y >= 0 and args.crop_w > 0 and args.crop_h > 0:
        crop = (args.crop_x, args.crop_y, args.crop_w, args.crop_h)

    try:
        extract_hardsub_ocr_srt(
            video_path=args.video_path,
            output_srt=args.output_srt,
            lang=args.lang,
            crop=crop,
            use_gpu=args.use_gpu,
            ocr_fps=args.ocr_fps,
            ocr_model=args.ocr_model
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
