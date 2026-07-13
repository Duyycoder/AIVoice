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

    # ---- Phân giải cookies (dùng chung cho MỌI lần tải bên dưới) ----
    resolved_cookiefile = None
    if cookies_file:
        abs_cookies_path = os.path.abspath(cookies_file)
        if not os.path.exists(abs_cookies_path):
            # __file__ = <repo>/AIVoice/apps/MediaComposer/app/services/video_downloader.py
            # -> 4x ".." = AIVoice root ; 5x ".." = repository root (parent of AIVoice)
            this_dir = os.path.dirname(__file__)
            repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "..", ".."))
            rel_repo_path = os.path.join(repo_root, cookies_file)
            aivoice_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", ".."))
            rel_aivoice_path = os.path.join(aivoice_root, cookies_file)
            if os.path.exists(rel_repo_path):
                abs_cookies_path = rel_repo_path
            elif os.path.exists(rel_aivoice_path):
                abs_cookies_path = rel_aivoice_path

        if os.path.exists(abs_cookies_path):
            resolved_cookiefile = abs_cookies_path
            log_json("download_info", {"message": f"Sử dụng file cookies tại: {abs_cookies_path}"})
        else:
            msg = f"Không tìm thấy file cookies tại: {cookies_file}"
            log_json("download_error", {"message": msg})
            raise FileNotFoundError(msg)

    def _base_opts():
        o = {
            'merge_output_format': 'mp4',
            'ffmpeg_location': ffmpeg_dir,
            'noplaylist': True,
            'quiet': True,
            'retries': 5,
            'fragment_retries': 5,
            'progress_hooks': [ytdl_hook],
        }
        if resolved_cookiefile:
            o['cookiefile'] = resolved_cookiefile
        return o

    def _download(fmt: str, tmpl: str):
        """Tải theo 1 format string, trả (đường dẫn tuyệt đối, id)."""
        opts = _base_opts()
        opts['format'] = fmt
        opts['outtmpl'] = os.path.join(output_dir, tmpl)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            fn = ydl.prepare_filename(info)
            base, _ = os.path.splitext(fn)
            mp4 = base + ".mp4"
            if not os.path.exists(mp4):
                mp4 = fn if os.path.exists(fn) else mp4
            if not os.path.exists(mp4):
                prefix = os.path.basename(base)
                for f in os.listdir(output_dir):
                    if f.startswith(prefix):
                        mp4 = os.path.join(output_dir, f)
                        break
            return os.path.abspath(mp4), info.get('id')

    def _has_audio(path: str) -> bool:
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            c = VideoFileClip(path)
            ok = c.audio is not None
            c.close()
            return ok
        except Exception as e:
            log_json("download_warning", {"message": f"Không kiểm tra được luồng audio ({e}) — coi như đã có."})
            return True  # không chắc chắn -> tránh tải lại thừa

    log_json("download_start", {"url": url, "platform": platform})
    try:
        # 1) Tải bản CHẤT LƯỢNG CAO NHẤT (có thể video-only trên TikTok h265 1080p).
        video_path, vid = _download('bv*+ba/b', 'dl_%(id)s.%(ext)s')

        # 2) Nếu đã có tiếng -> xong (YouTube/Bilibili/đa số trường hợp).
        if _has_audio(video_path):
            log_json("download_done", {"path": video_path})
            return video_path

        # 3) Video HQ không kèm tiếng (TikTok h265): tải luồng audio h264 riêng rồi ffmpeg mux
        #    -> giữ nguyên độ phân giải cao NHƯNG có tiếng. Loại h265/hevc khỏi nguồn audio vì
        #    các bản đó khai man có aac mà thực chất video-only.
        log_json("download_info", {"message": "Bản video HQ không kèm tiếng — đang tải luồng âm thanh riêng để ghép..."})
        try:
            audio_src, _ = _download('ba*[vcodec!*=h265][vcodec!*=hev]/b[vcodec!*=h265]', 'audiosrc_%(id)s.%(ext)s')
        except Exception as e:
            log_json("download_warning", {"message": f"Không tải được luồng âm thanh riêng ({e}) — trả video không tiếng."})
            log_json("download_done", {"path": video_path})
            return video_path

        if not _has_audio(audio_src):
            log_json("download_warning", {"message": "Nguồn âm thanh tách ra cũng không có tiếng — trả video không tiếng."})
            try:
                os.remove(audio_src)
            except Exception:
                pass
            log_json("download_done", {"path": video_path})
            return video_path

        import subprocess
        muxed_tmp = os.path.join(output_dir, f"dl_{vid}_av.mp4")
        cmd = [ffmpeg_bin, "-y", "-i", video_path, "-i", audio_src,
               "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", muxed_tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            os.remove(audio_src)
        except Exception:
            pass

        if r.returncode != 0:
            log_json("download_warning", {"message": f"Ghép video+audio thất bại — giữ video không tiếng: {r.stderr[-300:]}"})
            log_json("download_done", {"path": video_path})
            return video_path

        # Thay bản HQ-không-tiếng bằng bản đã ghép, giữ nguyên tên dl_<id>.mp4
        try:
            os.replace(muxed_tmp, video_path)
            final = video_path
        except Exception:
            final = muxed_tmp
        log_json("download_done", {"path": os.path.abspath(final)})
        return os.path.abspath(final)
    except Exception as e:
        log_json("download_error", {"error": str(e)})
        raise RuntimeError(f"Lỗi khi tải video: {e}")
