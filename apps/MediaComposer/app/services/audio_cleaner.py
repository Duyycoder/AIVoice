import os
import subprocess
import shutil
from loguru import logger

def isolate_vocals(input_audio_path: str, output_dir: str) -> str:
    """
    Sử dụng Demucs để tách giọng nói (vocals) ra khỏi tạp âm/nhạc nền.
    
    Args:
        input_audio_path (str): Đường dẫn tới file âm thanh gốc cần làm sạch.
        output_dir (str): Thư mục chứa kết quả đầu ra.
        
    Returns:
        str: Đường dẫn tới file âm thanh đã được làm sạch (chỉ chứa giọng nói),
             hoặc trả về input_audio_path nếu có lỗi xảy ra.
    """
    logger.info(f"Starting audio cleaning (vocal isolation) for: {input_audio_path}")
    
    try:
        # Sử dụng mô hình htdemucs mặc định, chỉ cần tách 2 stems: vocals và phần còn lại
        cmd = [
            "demucs",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-o", output_dir,
            input_audio_path
        ]
        
        logger.info(f"Running Demucs with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Demucs failed with error: {result.stderr}")
            return input_audio_path
            
        # File kết quả sẽ nằm trong thư mục <output_dir>/htdemucs/<tên_file_gốc_không_có_đuôi>/vocals.wav
        filename = os.path.splitext(os.path.basename(input_audio_path))[0]
        vocals_path = os.path.join(output_dir, "htdemucs", filename, "vocals.wav")
        
        if os.path.exists(vocals_path):
            logger.info(f"Audio cleaning completed successfully. Vocals saved to: {vocals_path}")
            return vocals_path
        else:
            logger.error(f"Demucs completed but vocals file not found at {vocals_path}")
            return input_audio_path
            
    except Exception as e:
        logger.exception(f"An error occurred during audio cleaning: {e}")
        return input_audio_path
