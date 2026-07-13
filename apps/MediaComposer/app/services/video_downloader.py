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
    
    # Append ffmpeg_dir and phantomjs_dir to PATH so yt-dlp can find ffmpeg and phantomjs.exe
    phantomjs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "phantomjs"))
    original_path = os.environ.get("PATH", "")
    paths_to_add = []
    if ffmpeg_dir not in original_path:
        paths_to_add.append(ffmpeg_dir)
    if phantomjs_dir not in original_path:
        paths_to_add.append(phantomjs_dir)
    if paths_to_add:
        os.environ["PATH"] = os.pathsep.join(paths_to_add) + os.pathsep + original_path
        
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
        # bv*+ba/b: ưu tiên ghép luồng video tốt nhất + audio tốt nhất (tránh tải nhầm bản
        # video-only như TikTok đôi khi trả về); fallback 'b' là file progressive tốt nhất.
        'format': 'bv*+ba/b',
        'outtmpl': os.path.join(output_dir, 'dl_%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_dir,
        'noplaylist': True,
        'quiet': True,
        'progress_hooks': [ytdl_hook],
    }
    
    if cookies_file:
        abs_cookies_path = os.path.abspath(cookies_file)
        if not os.path.exists(abs_cookies_path):
            # __file__ = <repo>/AIVoice/apps/MediaComposer/app/services/video_downloader.py
            # -> 4x ".." = AIVoice root ; 5x ".." = repository root (parent of AIVoice)
            this_dir = os.path.dirname(__file__)

            # Try resolving relative to repository root (parent of AIVoice) — vị trí ví dụ UI: configs/cookies_iqiyi.txt
            repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "..", ".."))
            rel_repo_path = os.path.join(repo_root, cookies_file)

            # Try resolving relative to AIVoice submodule root
            aivoice_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", ".."))
            rel_aivoice_path = os.path.join(aivoice_root, cookies_file)
            
            if os.path.exists(rel_repo_path):
                abs_cookies_path = rel_repo_path
            elif os.path.exists(rel_aivoice_path):
                abs_cookies_path = rel_aivoice_path
                
        if os.path.exists(abs_cookies_path):
            ydl_opts['cookiefile'] = abs_cookies_path
            log_json("download_info", {"message": f"Sử dụng file cookies tại: {abs_cookies_path}"})
        else:
            msg = f"Không tìm thấy file cookies tại: {cookies_file}"
            log_json("download_error", {"message": msg})
            raise FileNotFoundError(msg)
    
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
