import os
import sys
import json
import yt_dlp
from app.utils import utils

def log_json(event: str, data: dict):
    """Outputs progress log as a JSON string to stdout."""
    print(json.dumps({"event": event, **data}, ensure_ascii=False))
    sys.stdout.flush()


# ---- Làm sạch file cookies ------------------------------------------------
# Bản export cookies từ trình duyệt kèm theo token của WAF (_waftokenid). Token
# này gắn với dấu vân tay trình duyệt, gửi kèm từ yt-dlp (User-Agent khác) là bị
# TikTok trả 403 ngay ở bước tải trang — trong khi bỏ nó đi thì đăng nhập vẫn
# hiệu lực. Lọc ra 1 bản sao tạm, cũng để yt-dlp khỏi ghi đè file gốc của người
# dùng khi nó lưu lại cookie jar lúc kết thúc.

WAF_COOKIE_PREFIXES = ("_waf",)


def sanitize_cookies_file(path: str, work_dir: str):
    """Chép file cookies sang bản tạm, bỏ cookie WAF. Trả (đường dẫn bản tạm, tên đã bỏ).

    Trả (None, []) nếu không đọc/ghi được — khi đó dùng thẳng file gốc.
    """
    try:
        with open(path, encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()

        kept, dropped = [], []
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) == 7 and not (ln.startswith("#") and not ln.startswith("#HttpOnly_")):
                name = parts[5]
                if name.startswith(WAF_COOKIE_PREFIXES):
                    dropped.append(name)
                    continue
            kept.append(ln)

        if not dropped:
            return None, []

        os.makedirs(work_dir, exist_ok=True)
        out = os.path.join(work_dir, "_cookies_sanitized.txt")
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(kept) + "\n")
        return out, dropped
    except OSError as e:
        log_json("download_warning", {"message": f"Không lọc được file cookies ({e}) — dùng bản gốc."})
        return None, []


# ---- Chẩn đoán lỗi tải ----------------------------------------------------
# yt-dlp báo cùng một câu "No video formats found!" cho nhiều nguyên nhân rất
# khác nhau (tập phim TikTok Series, chặn IP, chống bot). Dịch sang câu người
# dùng hiểu được và biết làm gì tiếp.

COOKIES_HINT = ("khai báo file cookies (Netscape .txt) của tài khoản xem được video "
                "— Bước 4 → ô 'Cookies file', hoặc video.downloader_cookies trong Cấu hình")


def _is_tiktok(url: str, platform: str) -> bool:
    u = (url or "").lower()
    return platform in ("tiktok", "douyin") or "tiktok.com" in u or "douyin.com" in u


def _tiktok_drama_name(url: str, cookiefile: str = None):
    """Tên bộ phim nếu video là 1 tập TikTok Series, None nếu không phải/không rõ.

    Chạm vào nội bộ yt-dlp nên bọc try/except: bản yt-dlp mới có thể đổi API,
    khi đó chỉ mất câu chẩn đoán chi tiết chứ không hỏng luồng báo lỗi.
    """
    try:
        opts = {'quiet': True, 'no_warnings': True, 'simulate': True,
                'ignore_no_formats_error': True, 'socket_timeout': 15, 'retries': 0}
        if cookiefile:
            opts['cookiefile'] = cookiefile
        with yt_dlp.YoutubeDL(opts) as ydl:
            # Lượt này giải link rút gọn (vt.tiktok.com) và vượt JS challenge hộ.
            info = ydl.extract_info(url, download=False) or {}
            if info.get('formats'):
                return None
            ie = ydl.get_info_extractor('TikTok')
            ie.set_downloader(ydl)
            data, _status = ie._extract_web_data_and_status(
                info['webpage_url'], info['id'], fatal=False)
        return ((data or {}).get('dramaInfo') or {}).get('dramaName') or None
    except Exception:
        return None


def diagnose_download_failure(url: str, platform: str, err, cookiefile: str = None) -> str:
    """Dịch lỗi thô của yt-dlp thành câu nói rõ nguyên nhân + cách xử lý."""
    msg = str(err)
    low = msg.lower()

    if "403" in low and "forbidden" in low:
        return ("TikTok trả 403 ngay ở bước tải trang — gần như luôn do cookie WAF (_waftokenid) "
                "trong file cookies. Bản mới đã tự lọc cookie này; nếu vẫn 403 thì export lại "
                "cookies: đăng nhập TikTok, mở cửa sổ ẩn danh, export, rồi đóng cửa sổ đó ngay.")

    if "your ip address is blocked" in low:
        return ("TikTok chặn IP của bạn với video này (thường do giới hạn vùng). "
                f"Đổi mạng/VPN sang vùng xem được video, hoặc {COOKIES_HINT}.")

    if ("do not have permission" in low or "log into an account" in low
            or "login required" in low or "private" in low):
        return f"Video riêng tư hoặc yêu cầu đăng nhập. Hãy {COOKIES_HINT}."

    if "no video formats found" not in low:
        return msg

    if not _is_tiktok(url, platform):
        return (f"Không tìm thấy luồng video nào để tải — trang nguồn có thể yêu cầu "
                f"đăng nhập hoặc đã đổi cấu trúc. Hãy {COOKIES_HINT}. (lỗi gốc: {msg})")

    drama = _tiktok_drama_name(url, cookiefile)
    if drama:
        return (f"Video là 1 tập phim TikTok Series/mini-drama (\"{drama}\"). TikTok chỉ trả link "
                f"phát cho phiên ĐÃ ĐĂNG NHẬP và có quyền xem tập này; khách vãng lai bị bỏ trống "
                f"link. Hãy {COOKIES_HINT}. Nếu đã có cookies mà vẫn lỗi thì tài khoản đó chưa mở "
                f"khoá tập phim.")

    return ("TikTok trả về đủ thông tin video nhưng bỏ trống link phát (cơ chế chống bot). "
            f"Cách xử lý: (1) {COOKIES_HINT}; (2) nếu đã có cookies mà vẫn lỗi thì đợi vài "
            "phút rồi chạy lại — TikTok chặn tạm theo IP/phiên.")

def download_video(url: str, output_dir: str, platform: str = "generic", progress_cb=None, cookies_file: str = None) -> str:
    """Tải 1 video về output_dir, trả về đường dẫn file mp4. Raise nếu lỗi."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    ffmpeg_bin = utils.get_ffmpeg_binary()
    # KHÔNG dùng os.path.dirname(ffmpeg_bin): binary của imageio-ffmpeg tên là
    # ffmpeg-win-x86_64-v7.1.exe, yt-dlp không nhận ra nên báo "ffmpeg is not
    # installed" rồi bỏ bước ghép video+audio.
    ffmpeg_dir = utils.get_ffmpeg_dir_for_ytdlp()

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
    cleanup_paths = []
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
            sanitized, dropped = sanitize_cookies_file(abs_cookies_path, output_dir)
            if sanitized:
                resolved_cookiefile = sanitized
                cleanup_paths.append(sanitized)
                if dropped:
                    log_json("download_info", {"message": (
                        f"Đã bỏ {len(dropped)} cookie WAF ({', '.join(sorted(set(dropped)))}) "
                        f"khỏi bản sao — giữ lại sẽ bị TikTok trả 403.")})
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
        reason = diagnose_download_failure(url, platform, e, resolved_cookiefile)
        log_json("download_error", {"error": reason, "detail": str(e)})
        raise RuntimeError(reason)
    finally:
        for p in cleanup_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ---- Cào hàng loạt: giải link playlist/kênh thành danh sách video ----------
# Tải hàng loạt cần biết TRƯỚC có bao nhiêu video để hiện hàng đợi và cho người
# dùng cắt bớt. `extract_flat` chỉ đọc trang danh sách (không chạm từng video)
# nên nhanh và không tốn băng thông.

def _flat_entry(e: dict, fallback_url: str = "") -> dict:
    """Chuẩn hoá một mục yt-dlp (flat hoặc đầy đủ) về đúng thứ UI cần."""
    vid = e.get("id") or ""
    url = e.get("url") or e.get("webpage_url") or fallback_url
    # extract_flat trả 'url' là id trần với vài extractor -> dựng lại link đầy đủ.
    if url and not url.startswith("http"):
        url = e.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
    return {
        "id": vid,
        "url": url,
        "title": (e.get("title") or "").strip() or vid or "video",
        "duration": e.get("duration") or 0,
        "uploader": e.get("uploader") or e.get("channel") or "",
        "thumbnail": e.get("thumbnail") or "",
    }


def probe_entries(url: str, platform: str = "generic", cookies_file: str = None,
                  max_items: int = 0) -> list:
    """Trả danh sách video của một link.

    Link 1 video -> list 1 phần tử. Link playlist/kênh/hashtag -> mọi video bên
    trong (cắt còn `max_items` nếu > 0). Raise RuntimeError kèm câu chẩn đoán
    nếu không đọc được trang.
    """
    opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'skip_download': True,
        'socket_timeout': 30,
        'retries': 2,
    }
    if max_items and max_items > 0:
        opts['playlistend'] = int(max_items)
    if cookies_file and os.path.exists(cookies_file):
        opts['cookiefile'] = cookies_file

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as e:
        raise RuntimeError(diagnose_download_failure(url, platform, e, cookies_file))

    entries = info.get("entries")
    if entries is None:
        return [_flat_entry(info, url)]

    out = []
    for e in entries:
        if not e:
            continue
        # Kênh YouTube trả về playlist lồng playlist (Videos/Shorts/Live).
        if e.get("_type") == "playlist" and e.get("entries"):
            for sub in e["entries"]:
                if sub:
                    out.append(_flat_entry(sub))
        else:
            out.append(_flat_entry(e))
        if max_items and len(out) >= max_items:
            break
    return out
