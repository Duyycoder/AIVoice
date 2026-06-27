import os
from timeit import default_timer as timer
from loguru import logger
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from app.config import config
from app.utils import utils

model = None

def create_subtitle(audio_file, subtitle_file: str = ""):
    global model
    if WhisperModel is None:
        logger.warning("faster_whisper not available, skipping whisper subtitle generation")
        return ""
    
    model_size = config.whisper.get("model_size", "base")
    device = config.whisper.get("device", "cpu")
    compute_type = config.whisper.get("compute_type", "int8")
    
    if not model:
        logger.info(f"Loading faster-whisper model: {model_size} on {device}")
        try:
            model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)
        except Exception as e:
            logger.error(f"Failed to load whisper model: {e}")
            return None

    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    logger.info(f"Start transcription, output file: {subtitle_file}")
    
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(f"Detected language: '{info.language}', probability: {info.language_probability:.2f}")

    start = timer()
    subtitles = []

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            return
        subtitles.append({"msg": seg_text, "start_time": seg_start, "end_time": seg_end})

    for segment in segments:
        words_idx = 0
        words_len = len(segment.words)
        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                seg_text += word.word

                if utils.str_contains_punctuation(word.word):
                    seg_text = seg_text[:-1]
                    if not seg_text:
                        continue
                    recognized(seg_text, seg_start, seg_end)
                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if seg_text:
            recognized(seg_text, seg_start, seg_end)

    end = timer()
    logger.info(f"Transcription complete, elapsed: {end - start:.2f} s")

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(utils.text_to_srt(idx, text, subtitle.get("start_time"), subtitle.get("end_time")))
            idx += 1

    sub = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(sub)
    logger.info(f"Subtitle file created: {subtitle_file}")
    
    return subtitle_file

def read_srt_text(subtitle_file: str) -> str:
    """Reads an SRT file and returns just the text, concatenated."""
    if not os.path.exists(subtitle_file):
        return ""
    text_lines = []
    with open(subtitle_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            # Skip empty lines, indices, and timecodes
            if line and not line.isdigit() and "-->" not in line:
                text_lines.append(line)
    return " ".join(text_lines)

def create_subtitle_from_text(text: str, audio_duration: float, subtitle_file: str) -> str:
    """Creates an SRT subtitle file from plain text by distributing duration
    proportionally across sentences based on word count.

    This avoids loading Whisper entirely — zero GPU/CPU cost for transcription.
    Timing won't be word-level accurate, but is sufficient for stock video overlays.
    """
    import re
    if not text or not text.strip():
        logger.warning("Empty transcript text, skipping subtitle generation")
        return ""

    # Split into sentences by common sentence terminators
    raw_sentences = re.split(r'(?<=[.!?;:。！？；])\s+', text.strip())
    # Filter empty and merge very short fragments
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return ""

    # Calculate word count per sentence for proportional timing
    word_counts = [max(len(s.split()), 1) for s in sentences]
    total_words = sum(word_counts)

    # Small buffer at start/end to avoid edge-clipping
    MARGIN_SECONDS = 0.1
    usable_duration = max(audio_duration - MARGIN_SECONDS * 2, 1.0)

    lines = []
    current_time = MARGIN_SECONDS
    for idx, (sentence, wcount) in enumerate(zip(sentences, word_counts), start=1):
        proportion = wcount / total_words
        segment_duration = max(usable_duration * proportion, 0.3)
        start_time = current_time
        end_time = current_time + segment_duration
        lines.append(utils.text_to_srt(idx, sentence, start_time, end_time))
        current_time = end_time

    srt_content = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(srt_content)
    logger.info(f"Subtitle from text created: {subtitle_file} ({len(sentences)} segments)")
    return subtitle_file


def release_whisper_model():
    """Giải phóng mô hình faster-whisper khỏi bộ nhớ RAM/VRAM."""
    global model
    if model is not None:
        logger.info("Đang giải phóng mô hình Whisper khỏi bộ nhớ...")
        try:
            del model
        except Exception as e:
            logger.warning(f"Lỗi khi xóa tham chiếu mô hình: {e}")
        model = None
    
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Đã giải phóng bộ nhớ cache CUDA thành công.")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Lỗi khi dọn dẹp cache CUDA: {e}")

