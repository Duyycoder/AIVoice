import os
import re
import shutil
import hashlib
import json
from loguru import logger

def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def storage_dir(sub_dir="", create=False):
    d = os.path.join(root_dir(), "storage", sub_dir)
    if create and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return d

def font_dir():
    return os.path.join(root_dir(), "resource", "fonts")

def song_dir():
    return os.path.join(root_dir(), "resource", "songs")

def task_dir(task_id: str):
    d = os.path.join(storage_dir("tasks"), task_id)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return d

def get_ffmpeg_binary():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_ffmpeg_dir_for_ytdlp():
    """Trả về thư mục chứa binary tên đúng chuẩn 'ffmpeg' để đưa cho yt-dlp.

    imageio-ffmpeg đặt tên binary kèm phiên bản (ffmpeg-win-x86_64-v7.1.exe),
    trong khi yt-dlp chỉ chấp nhận file tên ffmpeg/ffprobe/avconv/avprobe. Trỏ
    thẳng vào thư mục của imageio-ffmpeg thì yt-dlp coi như KHÔNG có ffmpeg và
    bỏ luôn bước ghép video+audio ("You have requested merging of multiple
    formats but ffmpeg is not installed"). Vì vậy tạo sẵn một bản tên chuẩn
    (hardlink nếu cùng ổ đĩa, không thì copy) trong storage rồi trả thư mục đó.

    Nếu vì lý do nào đó không tạo được bản sao thì trả về thư mục gốc — lúc đó
    yt-dlp mất khả năng merge nhưng luồng tải vẫn chạy như trước.
    """
    src = get_ffmpeg_binary()
    if os.path.splitext(os.path.basename(src))[0].lower() == "ffmpeg":
        return os.path.dirname(src)

    dst_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    try:
        shim_dir = storage_dir("ffmpeg_bin", create=True)
        dst = os.path.join(shim_dir, dst_name)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return shim_dir

        tmp = f"{dst}.{os.getpid()}.tmp"
        try:
            os.link(src, tmp)
        except (OSError, AttributeError, NotImplementedError):
            shutil.copy2(src, tmp)
        try:
            os.replace(tmp, dst)
        except OSError:
            # Windows khoá file đang chạy: bản cũ vẫn dùng được thì giữ nguyên.
            os.remove(tmp)
            if not os.path.exists(dst):
                raise
        return shim_dir
    except OSError as e:
        logger.warning(f"Không tạo được bản ffmpeg tên chuẩn cho yt-dlp ({e}) — dùng thư mục gốc.")
        return os.path.dirname(src)

def parse_extension(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def text_to_srt(idx: int, msg: str, start_time: float, end_time: float) -> str:
    def format_time(t: float):
        h = int(t / 3600)
        m = int((t % 3600) / 60)
        s = int(t % 60)
        ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{idx}\n{format_time(start_time)} --> {format_time(end_time)}\n{msg}\n"

def str_contains_punctuation(word: str) -> bool:
    punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for char in word:
        if char in punctuation:
            return True
    return False

def to_json(obj) -> str:
    def handler(x):
        if hasattr(x, "__dict__"):
            return x.__dict__
        if hasattr(x, "dict") and callable(x.dict):
            return x.dict()
        return str(x)
    try:
        return json.dumps(obj, default=handler, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"<Serialization failed: {e}>"




# ---- Doc thong so video khong can ffprobe ---------------------------------
# May dich chi co ffmpeg.exe cua imageio-ffmpeg (khong kem ffprobe), con moviepy
# thi bo cuoc voi ffmpeg moi: no chay `ffmpeg -i <file>` roi coi ma thoat != 0 la
# loi, trong khi day chinh la cach ffmpeg in metadata ra stderr. Tu doc lay.

_RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RE_VIDEO_SIZE = re.compile(r"Stream #\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5})", re.DOTALL)
_RE_AUDIO = re.compile(r"Stream #\d+:\d+.*?Audio:")


def probe_media_info(path: str, timeout: float = 60.0) -> dict:
    """{width, height, duration, has_audio} cua mot file video.

    Tra ve dict rong neu khong chay duoc ffmpeg. Cac truong khong doc duoc thi
    bang 0/None de ben goi tu quyet dinh (vd: khong ro co audio -> dung coi la
    khong co, se tat nham long tieng).
    """
    import subprocess

    try:
        proc = subprocess.run(
            [get_ffmpeg_binary(), "-hide_banner", "-i", os.path.abspath(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return {}

    # `ffmpeg -i` khong co output file -> luon thoat khac 0, metadata nam o stderr.
    text = (proc.stderr or "") + (proc.stdout or "")
    if "Invalid data found" in text or "No such file" in text:
        return {}

    info = {"width": 0, "height": 0, "duration": 0.0, "has_audio": None}
    m = _RE_DURATION.search(text)
    if m:
        info["duration"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 2)
    m = _RE_VIDEO_SIZE.search(text)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    info["has_audio"] = bool(_RE_AUDIO.search(text))
    return info
