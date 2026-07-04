import os
import subprocess
import glob
import shutil
import gc
import sys

def get_ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return exe or "ffmpeg"

def get_ffprobe_exe():
    exe = shutil.which("ffprobe")
    if exe is None:
        ffmpeg_exe = get_ffmpeg_exe()
        if ffmpeg_exe and ffmpeg_exe != "ffmpeg":
            ffprobe_guess = ffmpeg_exe.lower().replace("ffmpeg", "ffprobe")
            if os.path.exists(ffprobe_guess):
                return ffprobe_guess
    return exe or "ffprobe"

def get_video_fps(video_path: str) -> str:
    """Uses cv2 to extract the framerate of the input video."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            # Format to 2 decimal places if it's not a whole number, otherwise integer
            return f"{fps:.2f}" if fps % 1 != 0 else str(int(fps))
        return "30"
    except Exception as e:
        print(f"Error getting FPS via cv2: {e}")
        return "30" # Default fallback

def extract_audio(input_video_path: str, output_audio_path: str) -> bool:
    """Extracts original audio from the video without re-encoding."""
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-i", input_video_path,
        "-vn",
        "-acodec", "copy",
        output_audio_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg audio extraction error: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}")
        return False
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy ffmpeg trong hệ thống.")
        return False

def clean_audio_with_demucs(input_audio_path: str, output_audio_path: str, temp_dir: str) -> bool:
    """Uses demucs to separate vocals from noise/music and returns the vocal track."""
    print("Sử dụng Demucs để làm sạch âm thanh (giữ lại giọng nói/vocals)...")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-o", temp_dir,
        input_audio_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Demucs will output to temp_dir/htdemucs/temp_audio/vocals.wav
        base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        vocal_path = os.path.join(temp_dir, "htdemucs", base_name, "vocals.wav")
        if os.path.exists(vocal_path):
            shutil.copy(vocal_path, output_audio_path)
            return True
        else:
            print(f"Demucs processing finished but {vocal_path} not found.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Demucs error: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}")
        return False

def extract_frames(input_video_path: str, output_frame_dir: str) -> bool:
    """Extracts all frames from the video as PNG images."""
    os.makedirs(output_frame_dir, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-i", input_video_path,
        "-qscale:v", "2",
        os.path.join(output_frame_dir, "frame_%06d.jpg")
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg frame extraction error: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}")
        return False
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy ffmpeg trong hệ thống.")
        return False

def apply_upscale_with_vram_cleanup(input_frame_dir: str, output_frame_dir: str, use_realesrgan: bool = False, tile_size: int = 512, log_callback=None):
    """
    Applies Real-ESRGAN upscale on all frames if use_realesrgan is True, otherwise simply copies them.
    Includes VRAM cleanup to prevent Out Of Memory errors.
    """
    os.makedirs(output_frame_dir, exist_ok=True)
    
    frames = sorted(glob.glob(os.path.join(input_frame_dir, "*.jpg")))
    total_frames = len(frames)
    
    if not use_realesrgan:
        if log_callback:
            log_callback("Upscale AI bị tắt. Đang sao chép trực tiếp khung hình...")
        for frame in frames:
            shutil.copy(frame, os.path.join(output_frame_dir, os.path.basename(frame)))
        return

    try:
        if log_callback:
            log_callback(f"Bắt đầu tải mô hình Real-ESRGAN...")
        
        import torch
        from PIL import Image
        import sys
        
        media_composer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "apps", "MediaComposer"))
        if media_composer_dir not in sys.path:
            sys.path.insert(0, media_composer_dir)
            
        from app.services.storytelling.postprocess import PostProcessor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        upscaler = PostProcessor(device=device, enable_upscaling=True, tile_size=tile_size)
        
        if log_callback:
            log_callback(f"Bắt đầu upscale {total_frames} khung hình bằng Real-ESRGAN trên {device.upper()}...")
        
        for i, frame in enumerate(frames):
            img = Image.open(frame).convert("RGB")
            
            # Upscale 4x directly without constraining to default 1920x1080, preserving aspect ratio
            upscaled_img = upscaler.run_realesrgan(img, scale=4, target_w=img.width * 4, target_h=img.height * 4)
            upscaled_img.save(os.path.join(output_frame_dir, os.path.basename(frame)), quality=95)
            
            if (i + 1) % 50 == 0 or (i + 1) == total_frames:
                if log_callback:
                    log_callback(f"Tiến độ Upscale AI: {i + 1}/{total_frames} khung hình...")
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
        if log_callback:
            log_callback("Đã hoàn thành Upscale AI, chuẩn bị ghép nối...")
        
    except Exception as e:
        if log_callback:
            log_callback(f"Cảnh báo: Lỗi khi chạy Real-ESRGAN ({e}). Fallback về copy frame gốc.")
        for frame in frames:
            shutil.copy(frame, os.path.join(output_frame_dir, os.path.basename(frame)))

def merge_video_and_audio(input_frame_dir: str, audio_path: str, output_video_path: str, fps: str = "30", resolution: str = "Gốc") -> bool:
    """Merges frames and audio into an MP4 using H.264 NVENC GPU acceleration."""
    # Build FFmpeg command to use NVIDIA GPU encoding
    if resolution == "720p (HD)":
        box = "1280:720"
    elif resolution == "1080p (Full HD)":
        box = "1920:1080"
    elif resolution == "1440p (2K)":
        box = "2560:1440"
    elif resolution == "2160p (4K)":
        box = "3840:2160"
    else:
        box = "4096:4096" # H.264 NVENC max limit
        
    vf_args = ["-vf", f"scale={box}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"]

    cmd = [
        get_ffmpeg_exe(), "-y",
        "-framerate", str(fps),
        "-i", os.path.join(input_frame_dir, "frame_%06d.jpg")
    ]
    if audio_path:
        cmd.extend(["-i", audio_path])
        
    cmd.extend(vf_args)
        
    cmd.extend([
        "-c:v", "h264_nvenc",
        "-preset", "p6",
        "-cq", "19",
        "-rc", "vbr",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_video_path
    ])
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg NVENC muxing error: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}")
        print("Falling back to CPU muxing (libx264)...")
        
        cmd_fallback = [
            get_ffmpeg_exe(), "-y",
            "-framerate", str(fps),
            "-i", os.path.join(input_frame_dir, "frame_%06d.jpg")
        ]
        if audio_path:
            cmd_fallback.extend(["-i", audio_path])
            
        cmd_fallback.extend(vf_args)
            
        cmd_fallback.extend([
            "-c:v", "libx264",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_video_path
        ])
        
        try:
            subprocess.run(cmd_fallback, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e2:
             print(f"FFmpeg CPU muxing error: {e2.stderr.decode('utf-8') if e2.stderr else 'Unknown error'}")
             return False

def process_animation_video(input_path: str, output_path: str, temp_dir: str, use_demucs: bool = False, use_realesrgan: bool = False, tile_size: int = 512, resolution: str = "Gốc", log_callback=None):
    """Main pipeline for the video sharpening workflow."""
    os.makedirs(temp_dir, exist_ok=True)
    
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
            
    audio_temp_path = os.path.join(temp_dir, "temp_audio.m4a")
    audio_clean_path = os.path.join(temp_dir, "temp_audio_clean.wav")
    frames_original_dir = os.path.join(temp_dir, "frames_original")
    frames_upscaled_dir = os.path.join(temp_dir, "frames_upscaled")
    
    try:
        log("Lấy cấu hình Video gốc...")
        fps = get_video_fps(input_path)
        
        log("Đang tách âm thanh...")
        if not extract_audio(input_path, audio_temp_path):
            # Try to continue if video has no audio
            log("Lưu ý: Video có thể không có âm thanh hoặc lỗi tách. Bỏ qua bước gộp âm thanh.")
            audio_temp_path = None
        elif use_demucs:
            log("Đang khử ồn/nhạc nền bằng Demucs...")
            if clean_audio_with_demucs(audio_temp_path, audio_clean_path, temp_dir):
                audio_temp_path = audio_clean_path
            else:
                log("Lỗi khi chạy Demucs, sử dụng lại âm thanh gốc.")
            
        log("Đang tách khung hình...")
        if not extract_frames(input_path, frames_original_dir):
            raise Exception("Tách khung hình thất bại.")
            
        log("Bắt đầu quy trình Upscale (Workflow 5 Trạm 3)...")
        apply_upscale_with_vram_cleanup(frames_original_dir, frames_upscaled_dir, use_realesrgan=use_realesrgan, tile_size=tile_size, log_callback=log)
        
        log(f"Đang mã hóa và gộp video (H264 NVENC, FPS: {fps})...")
        audio_to_merge = audio_temp_path if audio_temp_path and os.path.exists(audio_temp_path) else None
        
        if not merge_video_and_audio(frames_upscaled_dir, audio_to_merge, output_path, fps=fps, resolution=resolution):
            raise Exception("Gộp video và âm thanh thất bại.")
            
        log("Hoàn thành toàn bộ quy trình làm nét video.")
        return True
        
    except Exception as e:
        log(f"Lỗi trong quá trình xử lý video: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Dọn dẹp
        try:
            if audio_temp_path and os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
            if os.path.exists(frames_original_dir):
                shutil.rmtree(frames_original_dir)
            if os.path.exists(frames_upscaled_dir):
                shutil.rmtree(frames_upscaled_dir)
        except Exception:
            pass
