# -*- coding: utf-8 -*-
"""Tiện ích audio — đọc thời lượng CHÍNH XÁC, không bao giờ fallback âm thầm.

Bối cảnh (M1, 07/2026): pydub không tìm thấy ffmpeg trên máy user → đọc duration
fail → orchestrator fallback 60s ÂM THẦM → tách cảnh 10 phút audio thành 2 cảnh,
video bị cắt còn 60s. Module này vá tận gốc:
- .wav đọc bằng module `wave` chuẩn của Python (không cần ffmpeg).
- Định dạng khác dùng ffmpeg đóng gói sẵn trong imageio-ffmpeg (dự án đã có).
- Mọi đường thất bại → raise RuntimeError với thông báo rõ — KHÔNG đoán mò.
"""
import os
import re
import subprocess
from loguru import logger


def get_audio_duration(audio_path: str) -> float:
    """Trả thời lượng audio (giây). Raise RuntimeError nếu không đọc được.

    Thứ tự thử: wave (wav) → ffmpeg của imageio-ffmpeg → pydub (đã trỏ converter).
    """
    if not audio_path or not os.path.exists(audio_path):
        raise RuntimeError(f"File audio không tồn tại: {audio_path}")

    errors = []

    # 1) WAV: đọc native, nhanh và không phụ thuộc gì
    if audio_path.lower().endswith(".wav"):
        try:
            import wave
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0 and frames > 0:
                    dur = frames / float(rate)
                    logger.info(f"[AudioUtils] Duration (wave): {dur:.1f}s — {os.path.basename(audio_path)}")
                    return dur
        except Exception as e:
            errors.append(f"wave: {e}")

    # 2) ffmpeg từ imageio-ffmpeg (đã có sẵn trong requirements)
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [ffmpeg_exe, "-i", audio_path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
            errors="ignore",
        )
        # ffmpeg in "Duration: HH:MM:SS.xx" ra stderr
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
        if m:
            h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            dur = h * 3600 + mnt * 60 + s
            if dur > 0:
                logger.info(f"[AudioUtils] Duration (ffmpeg): {dur:.1f}s — {os.path.basename(audio_path)}")
                return dur
        errors.append("ffmpeg: không parse được Duration từ output")
    except Exception as e:
        errors.append(f"ffmpeg: {e}")

    # 3) pydub với converter trỏ về imageio-ffmpeg
    try:
        import imageio_ffmpeg
        from pydub import AudioSegment
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        audio = AudioSegment.from_file(audio_path)
        dur = len(audio) / 1000.0
        if dur > 0:
            logger.info(f"[AudioUtils] Duration (pydub): {dur:.1f}s — {os.path.basename(audio_path)}")
            return dur
    except Exception as e:
        errors.append(f"pydub: {e}")

    raise RuntimeError(
        f"Không đọc được thời lượng audio '{os.path.basename(audio_path)}'. "
        f"File có thể hỏng hoặc định dạng không hỗ trợ. Chi tiết: {' | '.join(errors)}"
    )
