import os
import re
from dataclasses import dataclass
from typing import List
from loguru import logger

from app.services.storytelling.models import Scene
from app.services.subtitle import create_subtitle

@dataclass
class SRTBlock:
    text: str
    start: float
    end: float

@dataclass
class SRTGroup:
    blocks: List[SRTBlock]
    start: float
    end: float

def parse_time(time_str: str) -> float:
    # 00:00:04,180
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].replace('.', ',').split(",")
    seconds = int(sec_parts[0])
    ms = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return hours * 3600 + minutes * 60 + seconds + ms / 1000.0

def parse_srt(srt_path: str) -> List[SRTBlock]:
    if not os.path.exists(srt_path):
        return []
        
    blocks = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    raw_blocks = re.split(r'\n\s*\n', content)
    for raw in raw_blocks:
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        if len(lines) >= 3:
            time_line = lines[1]
            if "-->" in time_line:
                t_parts = time_line.split("-->")
                start_t = parse_time(t_parts[0].strip())
                end_t = parse_time(t_parts[1].strip())
                text = " ".join(lines[2:])
                blocks.append(SRTBlock(text=text, start=start_t, end=end_t))
    return blocks

def map_scenes_to_timeline(
    scenes: List[Scene],
    srt_path: str = "",
    audio_path: str = "",
    silence_gap_threshold: float = 0.5,
    total_audio_duration: float = 0.0,
    use_whisper: bool = True,
    md_path: str = ""
) -> List[Scene]:
    """
    total_audio_duration: tổng thời lượng audio (giây).
    Dùng làm fallback nếu SRT không khả dụng.
    """
    if not scenes:
        return []

    if not srt_path or not os.path.exists(srt_path):
        if audio_path and os.path.exists(audio_path):
            if use_whisper:
                logger.info("No SRT provided, generating via Whisper...")
                from app.services.subtitle import create_subtitle, release_whisper_model
                srt_path = create_subtitle(audio_file=audio_path, language="vi")
                release_whisper_model()
                if not srt_path:
                    logger.warning("Failed to generate SRT via Whisper. Using uniform duration fallback.")
                    return _fallback_uniform_duration(scenes, total_audio_duration)
            else:
                logger.info("No-Whisper mode: generating subtitle based on word-count proportional timing from scenes.")
                from app.services.subtitle import utils
                
                word_counts = [max(len(s.text_vi.split()), 1) for s in scenes]
                total_words = sum(word_counts)
                
                MARGIN_SECONDS = 0.1
                usable_duration = max(total_audio_duration - MARGIN_SECONDS * 2, 1.0)
                
                current_time = MARGIN_SECONDS
                lines = []
                for idx, (scene, wcount) in enumerate(zip(scenes, word_counts), start=1):
                    proportion = wcount / total_words if total_words > 0 else 0
                    segment_duration = max(usable_duration * proportion, 0.3)
                    
                    scene.start_time = current_time
                    scene.end_time = current_time + segment_duration
                    scene.duration_sec = segment_duration
                    
                    lines.append(utils.text_to_srt(idx, scene.text_vi, scene.start_time, scene.end_time))
                    current_time = scene.end_time
                
                temp_dir = os.path.dirname(audio_path)
                srt_path = os.path.join(temp_dir, os.path.basename(audio_path) + ".srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                
                logger.info(f"Subtitle from scenes created: {srt_path} ({len(scenes)} segments)")
                return scenes
        else:
            logger.warning("No SRT and no Audio provided. Using uniform duration fallback.")
            return _fallback_uniform_duration(scenes, total_audio_duration)

    blocks = parse_srt(srt_path)
    if not blocks:
        logger.warning("Parsed SRT is empty. Using uniform duration fallback.")
        return _fallback_uniform_duration(scenes, total_audio_duration)

    # Nhóm SRTBlock
    groups: List[SRTGroup] = []
    current_group = []
    
    for block in blocks:
        if not current_group:
            current_group.append(block)
        else:
            last_end = current_group[-1].end
            if block.start - last_end > silence_gap_threshold:
                # Tạo group mới
                groups.append(SRTGroup(
                    blocks=current_group,
                    start=current_group[0].start,
                    end=current_group[-1].end
                ))
                current_group = [block]
            else:
                current_group.append(block)
                
    if current_group:
        groups.append(SRTGroup(
            blocks=current_group,
            start=current_group[0].start,
            end=current_group[-1].end
        ))
        
    num_scenes = len(scenes)
    num_groups = len(groups)
    
    if num_groups == 0:
        return _fallback_uniform_duration(scenes, total_audio_duration)

    # Map groups to scenes tuyến tính:
    # Nếu scenes > groups: scene thừa nhận fallback 5s
    # Nếu scenes < groups: scene cuối gộp toàn bộ groups còn lại
    group_idx = 0
    for i, scene in enumerate(scenes):
        if group_idx >= num_groups:
            # Scene thừa (nhiều scene hơn group SRT): fallback 5s liền sau scene trước
            prev_end = scenes[i - 1].end_time if i > 0 else 0.0
            scene.start_time = prev_end
            scene.end_time = prev_end + 5.0
            scene.duration_sec = 5.0
            logger.debug(f"Scene {i} has no matching SRT group, assigned 5s fallback.")
            continue

        scene.start_time = groups[group_idx].start

        if i == num_scenes - 1:
            # Scene cuối gộp toàn bộ groups còn lại
            scene.end_time = groups[-1].end
            group_idx = num_groups
        else:
            scene.end_time = groups[group_idx].end
            group_idx += 1

        scene.duration_sec = max(0.1, scene.end_time - scene.start_time)

    return scenes

def _fallback_uniform_duration(scenes: List[Scene], total_dur: float) -> List[Scene]:
    """
    Phân bổ đều thời gian cho các scene khi không có SRT.
    total_dur=0 → dùng 5s/scene làm an toàn tối thiểu.
    """
    if not scenes:
        return scenes
    effective_dur = total_dur if total_dur > 0 else len(scenes) * 5.0
    dur_per_scene = effective_dur / len(scenes)
    current = 0.0
    for scene in scenes:
        scene.start_time = current
        scene.end_time = current + dur_per_scene
        scene.duration_sec = dur_per_scene
        current = scene.end_time
    return scenes
