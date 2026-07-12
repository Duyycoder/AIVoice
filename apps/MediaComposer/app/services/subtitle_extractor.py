import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
import sys
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

def extract_hardsub_ocr_srt(
    video_path: str,
    output_srt: str,
    lang: str = "ch",                # "ch" (Trung) | "en" (Anh)
    crop: tuple = None,              # (x, y, w, h)
    time_start: str = "",            # "" hoặc "mm:ss"
    time_end: str = "",
    conf_threshold: int = 75,
    sim_threshold: int = 80,
    frames_to_skip: int = 1,
    use_gpu: bool = False,
    progress_cb=None
) -> str:
    """Gọi videocr-PaddleOCR: save_subtitles_to_file."""
    log_json("ocr_start", {"video_path": video_path, "output_srt": output_srt, "lang": lang})
    
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
            "use_gpu": use_gpu
        }
        
        if not use_fullframe:
            kwargs["crop_x"] = crop_x
            kwargs["crop_y"] = crop_y
            kwargs["crop_width"] = crop_width
            kwargs["crop_height"] = crop_height
            
        log_json("ocr_progress", {"message": "Đang chạy trích xuất phụ đề OCR (PaddleOCR)..."})
        save_subtitles_to_file(**kwargs)
        
        if not os.path.exists(output_srt) or os.path.getsize(output_srt) == 0:
            raise RuntimeError("OCR failed to generate subtitle file or generated file is empty.")
            
        log_json("ocr_done", {"output_srt": output_srt})
        return output_srt
    except Exception as e:
        log_json("ocr_error", {"error": str(e)})
        raise e
