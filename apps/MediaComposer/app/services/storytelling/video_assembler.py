import os
import subprocess
from dataclasses import dataclass
from typing import List
from loguru import logger

@dataclass
class FrameInfo:
    frame_path: str
    duration_sec: float

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
    
    work_dir = os.path.dirname(output_path)
    duration_txt = os.path.join(work_dir, "duration.txt")
    raw_video = os.path.join(work_dir, "raw_video.mp4")
    
    _create_duration_txt(frames, duration_txt)
    
    logger.info("Bước 1: Ghép ảnh → video thô")
    
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
        
    cmd1 = [
        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", duration_txt,
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        raw_video
    ]
    
    subprocess.run(cmd1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    logger.info("Bước 2: Mix audio + BGM + burn subtitle")
    
    # Thoát ký tự đặc biệt cho filter ffmpeg
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:") if srt_path else ""
    
    cmd2 = [ffmpeg_exe, "-y", "-i", raw_video, "-i", audio_path]
    
    if bgm_path and os.path.exists(bgm_path):
        cmd2.extend(["-i", bgm_path])
        filter_complex = f"[1:a]volume=1.0[voice];[2:a]volume={bgm_volume}[bgm];[voice][bgm]amix=inputs=2:duration=first[aout]"
        cmd2.extend(["-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]"])
    else:
        cmd2.extend(["-map", "0:v", "-map", "1:a"])
        
    if burn_subtitles and safe_srt and os.path.exists(srt_path):
        vf = f"subtitles='{safe_srt}':force_style='FontName=STHeitiMedium,FontSize=28'"
        cmd2.extend(["-vf", vf])
        
    total_dur = sum(f.duration_sec for f in frames)
    
    cmd2.extend([
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{total_dur:.3f}",
        output_path
    ])
    
    subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    if os.path.exists(duration_txt):
        os.remove(duration_txt)
    if os.path.exists(raw_video):
        os.remove(raw_video)
        
    logger.info(f"Video assembly completed: {output_path}")
    return output_path

def generate_draft_video(
    frames: List[FrameInfo],
    audio_path: str,
    srt_path: str,
    output_path: str
) -> str:
    
    work_dir = os.path.dirname(output_path)
    duration_txt = os.path.join(work_dir, "draft_duration.txt")
    
    _create_duration_txt(frames, duration_txt)
    
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:") if srt_path else ""
    vf = f"scale=512:910,subtitles='{safe_srt}':force_style='FontSize=14'" if safe_srt and os.path.exists(srt_path) else "scale=512:910"
    
    logger.info("Tạo draft video (không BGM, không upscale)...")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", duration_txt,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-shortest", "-map", "0:v", "-map", "1:a",
        output_path
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    if os.path.exists(duration_txt):
        os.remove(duration_txt)
        
    logger.info(f"Draft video completed: {output_path}")
    return output_path
