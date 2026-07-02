import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import List
from loguru import logger

from app.config import load_storytelling_config

@dataclass
class FrameInfo:
    frame_path: str
    duration_sec: float

def _get_valid_font(font_name: str) -> str:
    if not font_name:
        return "Arial"
    if os.path.exists(font_name):
        return font_name
    win_fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if os.path.exists(win_fonts_dir):
        try:
            fonts = os.listdir(win_fonts_dir)
            font_lower = font_name.lower().replace(" ", "")
            for f in fonts:
                f_name, _ = os.path.splitext(f)
                if font_lower in f_name.lower().replace(" ", ""):
                    return font_name
        except Exception:
            pass
    standard_fonts = {"arial", "tahoma", "calibri", "segoe ui", "times new roman", "verdana", "courier new", "microsoft sans serif"}
    if font_name.lower() in standard_fonts:
        return font_name
    
    logger.warning(f"Subtitle font '{font_name}' not found in Windows Fonts, falling back to 'Arial'")
    return "Arial"

def _create_duration_txt(frames: List[FrameInfo], output_txt: str) -> None:
    with open(output_txt, "w", encoding="utf-8") as f:
        for frame in frames:
            abs_path = os.path.abspath(frame.frame_path)
            safe_path = abs_path.replace("\\", "/")
            f.write(f"file '{safe_path}'\n")
            f.write(f"duration {frame.duration_sec:.3f}\n")
            
        if frames:
            abs_path = os.path.abspath(frames[-1].frame_path)
            safe_path = abs_path.replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

def assemble_video(
    frames: List[FrameInfo],
    audio_path: str,
    srt_path: str,
    output_path: str,
    bgm_path: str = "",
    bgm_volume: float = 0.15,
    burn_subtitles: bool = True
) -> str:
    
    work_dir = os.path.abspath(os.path.dirname(output_path))
    duration_txt = os.path.join(work_dir, "duration.txt")
    raw_video = os.path.join(work_dir, "raw_video.mp4")
    
    st_config = load_storytelling_config()
    out_w = st_config.get("output_width", 1920)
    out_h = st_config.get("output_height", 1080)
    
    _create_duration_txt(frames, duration_txt)
    
    logger.info("Bước 1: Ghép ảnh → video thô")
    
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
        
    cmd1 = [
        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", os.path.abspath(duration_txt),
        "-vf", f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-r", "25", "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        os.path.abspath(raw_video)
    ]
    
    subprocess.run(cmd1, check=True, cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    logger.info("Bước 2: Mix audio + BGM + burn subtitle")
    
    abs_audio = os.path.abspath(audio_path)
    abs_output = os.path.abspath(output_path)
    
    cmd2 = [ffmpeg_exe, "-y", "-i", os.path.abspath(raw_video), "-i", abs_audio]
    
    if bgm_path and os.path.exists(bgm_path):
        cmd2.extend(["-i", os.path.abspath(bgm_path)])
        filter_complex = f"[1:a]volume=1.0[voice];[2:a]volume={bgm_volume}[bgm];[voice][bgm]amix=inputs=2:duration=first[aout]"
        cmd2.extend(["-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]"])
    else:
        cmd2.extend(["-map", "0:v", "-map", "1:a"])
        
    temp_srt_path = os.path.join(work_dir, "temp_sub.srt")
    if burn_subtitles and srt_path and os.path.exists(srt_path):
        shutil.copyfile(srt_path, temp_srt_path)
        font_name = _get_valid_font(st_config.get("subtitle_font", "Arial"))
        font_size = st_config.get("subtitle_font_size", 28)
        vf = f"subtitles='temp_sub.srt':force_style='FontName={font_name},FontSize={font_size}'"
        cmd2.extend(["-vf", vf])
        
    total_dur = sum(f.duration_sec for f in frames)
    
    cmd2.extend([
        "-c:v", "libx264", "-r", "25", "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{total_dur:.3f}",
        abs_output
    ])
    
    subprocess.run(cmd2, check=True, cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    if os.path.exists(duration_txt):
        os.remove(duration_txt)
    if os.path.exists(raw_video):
        os.remove(raw_video)
    if os.path.exists(temp_srt_path):
        os.remove(temp_srt_path)
        
    logger.info(f"Video assembly completed: {output_path}")
    return output_path

def generate_draft_video(
    frames: List[FrameInfo],
    audio_path: str,
    srt_path: str,
    output_path: str
) -> str:
    
    st_config = load_storytelling_config()
    draft_w = st_config.get("image_width", 896)
    draft_h = st_config.get("image_height", 512)
    if draft_w % 2 != 0: draft_w -= 1
    if draft_h % 2 != 0: draft_h -= 1
    
    work_dir = os.path.abspath(os.path.dirname(output_path))
    duration_txt = os.path.join(work_dir, "draft_duration.txt")
    
    _create_duration_txt(frames, duration_txt)
    
    abs_audio = os.path.abspath(audio_path)
    abs_output = os.path.abspath(output_path)
    temp_srt_path = os.path.join(work_dir, "draft_temp_sub.srt")
    
    srt_exists = bool(srt_path and os.path.exists(srt_path))
    if srt_exists:
        shutil.copyfile(srt_path, temp_srt_path)
        font_name = _get_valid_font(st_config.get("subtitle_font", "Arial"))
        font_size = max(14, int(st_config.get("subtitle_font_size", 28) * 0.6))
        vf = f"scale={draft_w}:{draft_h},subtitles='draft_temp_sub.srt':force_style='FontName={font_name},FontSize={font_size}'"
    else:
        vf = f"scale={draft_w}:{draft_h}"
    
    logger.info("Tạo draft video (không BGM, không upscale)...")
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
        
    cmd = [
        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", os.path.abspath(duration_txt),
        "-i", abs_audio,
        "-vf", vf,
        "-c:v", "libx264", "-r", "25", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-shortest", "-map", "0:v", "-map", "1:a",
        abs_output
    ]
    
    subprocess.run(cmd, check=True, cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    if os.path.exists(duration_txt):
        os.remove(duration_txt)
    if os.path.exists(temp_srt_path):
        os.remove(temp_srt_path)
        
    logger.info(f"Draft video completed: {output_path}")
    return output_path

