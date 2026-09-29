import os
import sys
import wave
import json
import subprocess
import shutil
from loguru import logger
import soundfile as sf
import numpy as np

import torch

from app.utils import utils
# Ensure root AIVoice path is in sys.path to allow importing src.engines
project_root = os.path.dirname(os.path.dirname(utils.root_dir()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# If 'src' in sys.modules is the wrong one (from site-packages/namespace), remove it to force re-import
if 'src' in sys.modules:
    src_mod = sys.modules['src']
    is_wrong = False
    if getattr(src_mod, '__file__', None) is None:
        is_wrong = True
    elif 'site-packages' in str(src_mod.__file__):
        is_wrong = True
        
    if is_wrong:
        logger.info("Removing conflicting site-packages/namespace 'src' module from sys.modules.")
        del sys.modules['src']
        for k in list(sys.modules.keys()):
            if k.startswith('src.'):
                del sys.modules[k]

from app.config import config
from app.services.translation import parse_srt

def get_audio_duration(file_path: str) -> float:
    """Helper to get audio duration using standard wave module."""
    if not os.path.exists(file_path):
        return 0.0
    try:
        with wave.open(file_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / float(rate)
    except Exception as e:
        logger.warning(f"Failed to get duration using wave for {file_path}, trying soundfile: {e}")
        try:
            info = sf.info(file_path)
            return info.duration
        except Exception as sf_err:
            logger.error(f"Failed to get audio duration: {sf_err}")
            return 0.0

def extract_voice_sample(audio_path: str, srt_path: str, output_ref_path: str) -> bool:
    """
    Parses original SRT, finds a clear speech segment of 6-12s, 
    and slices it from audio_path using FFmpeg to serve as a reference audio for XTTSv2 voice cloning.
    """
    if not os.path.exists(srt_path):
        logger.warning("Original SRT file not found for voice sample extraction.")
        return False
        
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        srt_content = f.read()
        
    segments = parse_srt(srt_content)
    if not segments:
        logger.warning("No segments found in original SRT.")
        return False
        
    # Helper to parse timestamp to seconds
    def time_to_seconds(t_str: str) -> float:
        try:
            parts = t_str.split("-->")[0].strip().split(":")
            h = int(parts[0])
            m = int(parts[1])
            s_parts = parts[2].split(",")
            s = int(s_parts[0])
            ms = int(s_parts[1])
            return h * 3600 + m * 60 + s + ms / 1000.0
        except Exception:
            return 0.0

    def get_duration(seg):
        try:
            parts = seg["timestamp"].split("-->")
            start = time_to_seconds(parts[0].strip())
            end = time_to_seconds(parts[1].strip())
            return start, end, end - start
        except Exception:
            return 0.0, 0.0, 0.0

    # Look for a segment between 6.0 and 12.0 seconds with speech
    target_seg = None
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start, end, dur = get_duration(seg)
        if 6.0 <= dur <= 12.0:
            target_seg = (start, end)
            break
            
    # Fallback to 4.0 - 15.0 seconds
    if not target_seg:
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            start, end, dur = get_duration(seg)
            if 4.0 <= dur <= 15.0:
                target_seg = (start, end)
                break
                
    # Fallback to longest segment
    if not target_seg:
        longest_dur = 0.0
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            start, end, dur = get_duration(seg)
            if dur > longest_dur:
                longest_dur = dur
                target_seg = (start, end)
                
    if not target_seg:
        logger.warning("No suitable voice sample segment found.")
        return False
        
    start_time, end_time = target_seg
    logger.info(f"Extracting voice sample from segment: {start_time:.2f}s to {end_time:.2f}s")
    
    ffmpeg_bin = utils.get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{start_time:.3f}",
        "-to", f"{end_time:.3f}",
        "-i", audio_path,
        "-c", "copy",
        output_ref_path
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(output_ref_path) and os.path.getsize(output_ref_path) > 0:
        logger.success(f"Voice sample extracted successfully to {output_ref_path}")
        return True
    else:
        logger.error(f"Failed to extract voice sample: {res.stderr}")
        return False

def shorten_text_via_gemini(text: str, target_duration: float) -> str:
    """[DEPRECATED — không còn được gọi] Trước đây rút gọn lời thoại để vừa thời lượng,
    nhưng làm audio lệch chữ so với phụ đề đang hiển thị. Giữ lại phòng khi cần bật lại
    dạng tuỳ chọn. Fit thời lượng hiện do atempo (Rule 2 trong generate_dubbed_audio) lo."""
    api_key = config.app.get("openai_api_key")
    if not api_key:
        logger.warning("No API key configured for Gemini text shortening. Returning original text.")
        return text
        
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=config.app.get("openai_base_url", "https://api.openai.com/v1")
    )
    model = config.app.get("openai_model", "gemini-2.0-flash")
    
    prompt = f"""Bạn là một chuyên gia hiệu chỉnh phụ đề và thuyết minh video. Câu tiếng Việt sau đây quá dài và không thể đọc kịp trong khoảng thời gian {target_duration:.1f} giây.
Hãy viết lại câu này cực kỳ ngắn gọn, súc tích nhưng vẫn giữ nguyên ý nghĩa cốt lõi để người thuyết minh có thể đọc kịp.
Câu gốc: "{text}"

Chỉ trả về câu đã rút gọn, không thêm bất kỳ giải thích nào, không thêm dấu nháy kép bọc ngoài.
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        shortened = response.choices[0].message.content.strip()
        # Clean any quotes
        if shortened.startswith('"') and shortened.endswith('"'):
            shortened = shortened[1:-1].strip()
        logger.info(f"Gemini shortened text: '{text}' -> '{shortened}'")
        return shortened
    except Exception as e:
        logger.error(f"Failed to shorten text via Gemini: {e}")
        return text

def cat_khoang_lang(path: str, nguong: float = 0.01, chua_ms: int = 40) -> None:
    """Cắt lặng đầu/cuối file TTS (Edge/Kokoro thêm ~0,1–0,4 s) — lời bắt đầu đúng lúc phụ đề hiện, đo thời lượng thật."""
    data, sr = sf.read(path)
    mono = np.abs(data if data.ndim == 1 else data.mean(axis=1))
    co_tieng = np.nonzero(mono > nguong)[0]
    if not len(co_tieng):
        return
    chua = int(sr * chua_ms / 1000)
    dau, cuoi = max(0, co_tieng[0] - chua), min(len(data), co_tieng[-1] + chua)
    if dau > 0 or cuoi < len(data):
        sf.write(path, data[dau:cuoi], sr)


def cua_so_cau(moc: list, i: int, het_video: float) -> float:
    """Thời gian câu i được phép đọc: tới lúc câu SAU bắt đầu (không chỉ tới lúc phụ đề câu này tắt)."""
    bat_dau, ket_thuc = moc[i]
    sau = moc[i + 1][0] if i + 1 < len(moc) else het_video
    return max(ket_thuc, sau) - bat_dau


TUA_TOI_DA = 1.7       # atempo > ~1,7 nghe méo; vẫn dài hơn thì cắt đuôi (fade) — KHÔNG chồng lên câu sau
TRE_TOI_DA = 0.8       # giây một câu được đọc lấn sang chỗ câu sau (câu sau lùi theo) trước khi phải cắt đuôi


def xep_loi(cac_cau: list, tre_toi_da: float = TRE_TOI_DA) -> list:
    """[(bắt_đầu, hết_chỗ, dài)] (giây, theo thứ tự) → [(đặt_lúc, giữ_dài)]. Không bao giờ chồng 2 giọng; câu dài được lấn
    tối đa `tre_toi_da` s sang chỗ câu sau (câu sau lùi theo, bắt kịp lại ở khoảng nghỉ). Thoại dày (video game, đo 29/09):
    chỉ cắt đuôi ở cửa sổ gốc thì 17/25 câu mất chữ."""
    kq, con_tro = [], 0.0
    for bat_dau, het_cho, dai in cac_cau:
        dat = max(bat_dau, con_tro)
        giu = max(0.0, min(dai, het_cho + tre_toi_da - dat))
        kq.append((dat, giu))
        con_tro = dat + giu
    return kq


def generate_dubbed_audio(
    task_id: str,
    video_path: str,
    translated_srt_path: str,
    source_srt_path: str,
    engine_name: str,
    voice_name: str,
    ducking_ratio: float,
    auto_clone: bool = False
) -> str:
    """
    Core dubbing pipeline:
    1. Extract audio.
    2. Extract voice clone sample (if auto_clone).
    3. Loop through translated SRT segments, run TTS, adjust speed, and shorten if necessary.
    4. Compile segment audios into a silent timeline track.
    5. Perform Dynamic Audio Ducking and mixing via FFmpeg.
    """
    task_dir = utils.task_dir(task_id)
    ffmpeg_bin = utils.get_ffmpeg_binary()
    
    # 1. Extract audio from video
    extracted_audio = os.path.join(task_dir, "orig_audio.wav")
    logger.info("Extracting original audio track for dubbing processing...")
    cmd_extract = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        extracted_audio
    ]
    res_extract = subprocess.run(cmd_extract, capture_output=True, text=True)
    if res_extract.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {res_extract.stderr}")
        
    # Get total video duration
    from moviepy.video.io.VideoFileClip import VideoFileClip
    try:
        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        clip.close()
    except Exception:
        video_duration = get_audio_duration(extracted_audio)
        
    # 2. Handle Auto-Voice Cloning
    ref_audio_path = ""
    if engine_name == "clone":
        if auto_clone:
            ref_audio_path = os.path.join(task_dir, "auto_ref_voice.wav")
            success = extract_voice_sample(extracted_audio, source_srt_path, ref_audio_path)
            if not success:
                logger.warning("Auto clone sample extraction failed. Falling back to default ref_audio if available.")
                ref_audio_path = config.app.get("ref_audio", "")
        else:
            # Fallback to configured global ref audio
            ref_audio_path = config.app.get("ref_audio", "")
            
    # Ensure correct sys.path at runtime (module-level cleanup handles sys.modules)
    project_root = os.path.dirname(os.path.dirname(utils.root_dir()))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 3. Instantiate chosen TTS Engine
    engine = None
    logger.info(f"Initializing AIVoice TTS Engine: {engine_name}...")
    
    if engine_name == "edge":
        from src.engines.edge import EdgeEngine
        engine = EdgeEngine(voice=voice_name)
    elif engine_name == "piper":
        from src.engines.piper import PiperEngine
        model_path = os.path.abspath(os.path.join(project_root, "models", "piper", voice_name))
        engine = PiperEngine(model_path=model_path)
    elif engine_name == "kokoro":
        from src.engines.kokoro import KokoroEngine
        engine = KokoroEngine()
    elif engine_name == "vieneu":
        from src.engines.vieneu import VieNeuEngine
        engine = VieNeuEngine()
    elif engine_name == "clone":
        from src.engines.clone import CloneEngine
        model_path = os.path.abspath(os.path.join(project_root, "models", "xtts_v2"))
        engine = CloneEngine(model_path=model_path)
    else:
        raise ValueError(f"Unsupported TTS Engine: {engine_name}")
        
    # 4. Parse translated SRT segments
    with open(translated_srt_path, "r", encoding="utf-8", errors="ignore") as f:
        srt_content = f.read()
    segments = parse_srt(srt_content)
    
    if not segments:
        raise ValueError("No segments found in the translated subtitle file.")
        
    # Helper to parse timestamp to seconds
    def timestamp_to_seconds(t_str: str) -> float:
        try:
            parts = t_str.split(":")
            h = int(parts[0])
            m = int(parts[1])
            s_parts = parts[2].split(",")
            s = int(s_parts[0])
            ms = int(s_parts[1])
            return h * 3600 + m * 60 + s + ms / 1000.0
        except Exception:
            return 0.0
            
    temp_wavs = []
    moc = []
    for seg in segments:
        ts_parts = seg["timestamp"].split("-->")
        moc.append((timestamp_to_seconds(ts_parts[0].strip()), timestamp_to_seconds(ts_parts[1].strip())))
    so_cau_cat = 0

    # 5. Generate TTS for each segment and apply alignment rules
    logger.info(f"Generating TTS speech for {len(segments)} segments...")
    for idx, seg in enumerate(segments):
        if idx % 10 == 0 or idx == len(segments) - 1:
            percent = 10 + int((idx / max(1, len(segments) - 1)) * 80)
            # KHÔNG import json/sys cục bộ ở đây: hàm đã dùng sys.path phía trên → UnboundLocalError (agy lượt 24).
            print(json.dumps({"event": "autosub_progress", "message": f"Đang đọc câu {idx + 1}/{len(segments)}", "percent": percent}, ensure_ascii=False))
            sys.stdout.flush()
            
        start_time, end_time = moc[idx]
        # Được đọc tới lúc câu sau bắt đầu: trước 29/09 chỉ tính tới lúc phụ đề tắt → tua nhanh cả câu vẫn còn chỗ.
        target_dur = cua_so_cau(moc, idx, video_duration)
        
        raw_text = seg["text"].strip()
        if not raw_text:
            continue
            
        temp_seg_raw = os.path.join(task_dir, f"seg_{idx}_raw.wav")
        temp_seg_fitted = os.path.join(task_dir, f"seg_{idx}_fitted.wav")
        
        # Helper function to generate WAV via TTS
        def run_tts(text_to_speak, path):
            if engine_name == "edge":
                engine.generate(text_to_speak, path, voice=voice_name)
            elif engine_name == "piper":
                engine.generate(text_to_speak, path, voice=voice_name)
            elif engine_name == "kokoro":
                engine.generate(text_to_speak, path, voice=voice_name)
            elif engine_name == "vieneu":
                # Defensive fallback for UI caching
                if voice_name in ["v3turbo", "standard"]:
                    v_name = "Ngọc Lan"
                    v_mode = voice_name
                elif voice_name and "|" in voice_name:
                    v_name, v_mode = voice_name.split("|")
                else:
                    v_name = "Ngọc Lan"
                    v_mode = "v3turbo"
                engine.generate(text_to_speak, path, voice=v_name, vieneu_mode=v_mode)
            elif engine_name == "clone":
                # XTTSv2 Zero Shot cloning needs ref_audio
                engine.generate(text_to_speak, path, ref_audio=ref_audio_path, voice="vi")
                
        # Generate initial audio
        run_tts(raw_text, temp_seg_raw)
        
        if not os.path.exists(temp_seg_raw) or os.path.getsize(temp_seg_raw) == 0:
            logger.warning(f"Failed to generate TTS for segment {idx}. Skipping.")
            continue

        try:
            cat_khoang_lang(temp_seg_raw)
        except Exception as e:
            logger.warning(f"Segment {idx}: không cắt được khoảng lặng: {e}")
        actual_dur = get_audio_duration(temp_seg_raw)
        logger.info(f"Segment {idx}: Target duration = {target_dur:.2f}s, TTS actual = {actual_dur:.2f}s")
        
        # Rule 1 (Gemini text-shortening) ĐÃ BỎ: giữ lời thoại khớp nguyên văn phụ đề để
        # audio không lệch chữ so với sub (dubbing chỉ đọc đúng câu đang hiển thị).
        # Việc fit thời lượng do Rule 2 (tua tốc độ) đảm nhiệm — không đổi chữ.
        tolerance = 0.05     # cửa sổ đã tính tới câu sau → không cần nới thêm (0,3 cũ làm lời tràn sang câu sau)

        # Rule 2: Time-stretching/Speeding up if it still exceeds target by more than the tolerance
        if target_dur > 0 and actual_dur > target_dur + tolerance:
            factor = actual_dur / target_dur
            factor = min(factor, TUA_TOI_DA)
            logger.info(f"Segment {idx}: Speeding up by {factor:.2f}x to fit target duration...")
            
            # FFmpeg atempo speedup without changing pitch
            cmd_speed = [
                ffmpeg_bin,
                "-y",
                "-i", temp_seg_raw,
                "-filter:a", f"atempo={factor:.3f}",
                temp_seg_fitted
            ]
            res_speed = subprocess.run(cmd_speed, capture_output=True, text=True)
            if res_speed.returncode != 0:
                logger.error(f"FFmpeg speedup failed: {res_speed.stderr}")
                shutil.copy2(temp_seg_raw, temp_seg_fitted)
            else:
                # Clean up raw temp
                os.remove(temp_seg_raw)
        else:
            # Fits perfectly or is within tolerance, just copy to fitted path
            shutil.move(temp_seg_raw, temp_seg_fitted)
            
        temp_wavs.append({
            "index": idx,
            "start_time": start_time,
            "end_time": end_time,
            "duration": target_dur,
            "file_path": temp_seg_fitted
        })
        
    # Free TTS model memory
    if hasattr(engine, "tts"):
        logger.info("Cleaning up PyTorch/TTS model memory...")
        del engine.tts
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    if not temp_wavs:
        raise RuntimeError("No voiceover segments could be generated.")
        
    # 6. Timeline Assembly (Căn chỉnh trục thời gian)
    logger.info("Assembling dubbed segments into continuous timeline track...")
    
    # Read sample rate from first segment to ensure consistency
    first_info = sf.info(temp_wavs[0]["file_path"])
    sample_rate = first_info.samplerate
    
    # Create silent timeline array
    total_samples = int(sample_rate * video_duration)
    voiceover_array = np.zeros(total_samples, dtype=np.float32)
    
    cac_loi = []
    for item in temp_wavs:
        data, sr = sf.read(item["file_path"])
        # Downmix to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        if sr != sample_rate:
            # Engine khác tần số mẫu (vd Clone 24k vs 22,05k): đặt sai tần số làm câu lệch dần về cuối video.
            data = np.interp(np.linspace(0, len(data), int(len(data) * sample_rate / sr), endpoint=False),
                             np.arange(len(data)), data)
        cac_loi.append(data)
    vi_tri = xep_loi([(it["start_time"], it["start_time"] + it["duration"], len(d) / sample_rate)
                      for it, d in zip(temp_wavs, cac_loi)])

    for item, data, (dat_luc, giu_dai) in zip(temp_wavs, cac_loi, vi_tri):
        start_sample = int(dat_luc * sample_rate)
        # Không cho lời tràn sang câu sau (trước đây CỘNG chồng → 2 giọng cùng lúc): quá chỗ cho phép thì cắt đuôi + fade 80 ms.
        toi_da = int(giu_dai * sample_rate)
        if len(data) > toi_da:
            data = data[:max(1, toi_da)].copy()
            fade = min(len(data), int(0.08 * sample_rate))
            data[-fade:] *= np.linspace(1.0, 0.0, fade)
            so_cau_cat += 1
        end_sample = start_sample + len(data)

        if end_sample > total_samples:
            end_sample = total_samples
            data = data[:end_sample - start_sample]
            
        # Add audio samples onto the timeline
        voiceover_array[start_sample:end_sample] += data
        
    if so_cau_cat:
        logger.warning(f"{so_cau_cat}/{len(temp_wavs)} câu đọc không kịp dù tua {TUA_TOI_DA}x + lấn {TRE_TOI_DA}s — đã cắt đuôi. "
                       "Rút gọn các câu này trong phụ đề hoặc tăng tốc độ giọng.")

    # Save final assembled voiceover track
    voiceover_track_path = os.path.join(task_dir, "vietnamese_voiceover_track.wav")
    sf.write(voiceover_track_path, voiceover_array, sample_rate)
    logger.success(f"Assembled voiceover track saved: {voiceover_track_path}")
    
    # Clean up individual segment files
    for item in temp_wavs:
        if os.path.exists(item["file_path"]):
            os.remove(item["file_path"])
            
    # 7. Dynamic Audio Ducking and mixing via FFmpeg script
    logger.info("Performing Dynamic Audio Ducking and Mixing...")
    duck_volume = max(0.0, 1.0 - (ducking_ratio / 100.0))
    logger.info(f"Ducking volume multiplier: {duck_volume:.2f} (reduction ratio: {ducking_ratio}%)")
    
    # Generate the filtergraph between expressions for ducking
    intervals = []
    for item in temp_wavs:
        # Pad ducking window slightly by 100ms at start/end for smooth audio transition
        s = max(0.0, item["start_time"] - 0.1)
        e = min(video_duration, item["end_time"] + 0.1)
        intervals.append(f"between(t,{s:.3f},{e:.3f})")
        
    enable_expr = "+".join(intervals)
    
    # Write filtergraph to a file to bypass Windows command line character limits
    filtergraph_path = os.path.join(task_dir, "ducking_filter.txt")
    
    # Filter explanation:
    # [0:a] is the original audio. We apply volume filter: if 'enable_expr' is True, volume is duck_volume, else 1.0.
    # [1:a] is the voiceover track. We mix them together using 'amix'.
    filter_content = f"[0:a]volume=enable='{enable_expr}':volume={duck_volume:.3f}[a0];[a0][1:a]amix=inputs=2:duration=first:dropout_transition=0"
    
    with open(filtergraph_path, "w", encoding="utf-8") as f:
        f.write(filter_content)
        
    mixed_audio_path = os.path.join(task_dir, "dubbed_mixed_audio.wav")
    
    # Run FFmpeg mix script
    cmd_mix = [
        ffmpeg_bin,
        "-y",
        "-i", extracted_audio,
        "-i", voiceover_track_path,
        "-filter_complex_script", filtergraph_path,
        "-ac", "2",  # output stereo
        mixed_audio_path
    ]
    
    logger.info(f"Running FFmpeg complex mix script...")
    res_mix = subprocess.run(cmd_mix, capture_output=True, text=True)
    if res_mix.returncode != 0:
        logger.error(f"FFmpeg mixing failed: {res_mix.stderr}")
        raise RuntimeError(f"FFmpeg audio mixing failed: {res_mix.stderr}")
        
    logger.success(f"Dynamic Audio Ducking complete: {mixed_audio_path}")
    return mixed_audio_path
