import os
import sys
import json
import yt_dlp
from app.utils import utils

def log_json(event: str, data: dict):
    """Outputs progress log as a JSON string to stdout."""
    print(json.dumps({"event": event, **data}, ensure_ascii=False))
    sys.stdout.flush()

def download_video(url: str, output_dir: str, platform: str = "generic", progress_cb=None, cookies_file: str = None) -> str:
    """Tải 1 video về output_dir, trả về đường dẫn file mp4. Raise nếu lỗi."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    ffmpeg_bin = utils.get_ffmpeg_binary()
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    
    # Progress hook for yt-dlp
    def ytdl_hook(d):
        if d.get('status') == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            percent = (downloaded / total) * 100 if total else 0
            speed = d.get('speed') or 0
            eta = d.get('eta') or 0
            
            # Format speed nicely (e.g. KB/s, MB/s)
            speed_str = ""
            if speed > 1024 * 1024:
                speed_str = f"{speed / (1024 * 1024):.2f} MB/s"
            elif speed > 1024:
                speed_str = f"{speed / 1024:.2f} KB/s"
            else:
                speed_str = f"{speed} B/s"
                
            progress_data = {
                "percent": round(percent, 2),
                "speed": speed_str,
                "eta": eta
            }
            if progress_cb:
                progress_cb(progress_data)
            else:
                log_json("download_progress", progress_data)
        elif d.get('status') == 'finished':
            if progress_cb:
                progress_cb({"percent": 100.0, "status": "finished"})
            else:
                log_json("download_progress", {"percent": 100.0, "status": "finished"})

    # Setup YoutubeDL options
    ydl_opts = {
        'format': 'mp4/best',
        'outtmpl': os.path.join(output_dir, 'dl_%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_dir,
        'noplaylist': True,
        'quiet': True,
        'progress_hooks': [ytdl_hook],
    }
    
    if cookies_file:
        abs_cookies_path = os.path.abspath(cookies_file)
        if os.path.exists(abs_cookies_path):
            ydl_opts['cookiefile'] = abs_cookies_path
        else:
            log_json("download_warning", {"message": f"Không tìm thấy file cookies tại: {cookies_file}"})
    
    log_json("download_start", {"url": url, "platform": platform})
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            # Ensure the output filename has .mp4 extension
            base, _ = os.path.splitext(filename)
            mp4_filename = base + ".mp4"
            
            if not os.path.exists(mp4_filename) and os.path.exists(filename):
                os.rename(filename, mp4_filename)
                
            if not os.path.exists(mp4_filename):
                # Look for downloaded files in output_dir
                files = os.listdir(output_dir)
                for f in files:
                    if f.startswith(f"dl_{info.get('id')}"):
                        mp4_filename = os.path.join(output_dir, f)
                        break
                        
            log_json("download_done", {"path": mp4_filename})
            return os.path.abspath(mp4_filename)
        except Exception as e:
            log_json("download_error", {"error": str(e)})
            raise RuntimeError(f"Lỗi khi tải video: {e}")
