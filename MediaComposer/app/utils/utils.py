import os
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

