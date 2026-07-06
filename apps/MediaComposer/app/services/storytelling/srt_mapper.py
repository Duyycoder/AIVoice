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


def map_semantic_scenes_to_srt(scenes, srt_blocks, total_duration):
    """Map semantic scenes (list of Scene dataclass) sang timeline bằng text matching.

    Thuật toán:
    1. Chuẩn hoá text 2 phía: lowercase, bỏ dấu câu, NFKD bỏ dấu tiếng Việt
    2. Ghép SRT blocks thành 1 chuỗi từ, ghi (block_index, word_position)
    3. Từng cảnh: difflib.SequenceMatcher tìm đoạn khớp (tuyến tính, không quay lui)
    4. Ratio < 0.5 → gán theo tỷ lệ số từ + log warning
    5. Bảo đảm: start/end tăng dần, không chồng lấn, cảnh cuối = total_duration
    """
    import unicodedata
    import difflib

    if not scenes:
        return scenes

    # Nếu không có SRT blocks → fallback tỷ lệ từ
    if not srt_blocks:
        logger.info("[SemanticSRT] No SRT blocks, using word-count proportional timing.")
        return _fallback_uniform_duration(scenes, total_duration)

    def normalize_text(text):
        """Lowercase, bỏ dấu tiếng Việt, bỏ dấu câu."""
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text

    # Ghép SRT thành chuỗi từ liên tục + map vị trí từ → block index
    srt_words = []
    word_to_block = []  # index trong srt_words → index trong srt_blocks
    for blk_idx, block in enumerate(srt_blocks):
        words = normalize_text(block.text).split()
        for w in words:
            srt_words.append(w)
            word_to_block.append(blk_idx)

    srt_full = " ".join(srt_words)
    cursor = 0  # vị trí từ hiện tại (tuyến tính, không quay lui)

    total_scene_words = sum(len(s.text_vi.split()) for s in scenes)
    if total_scene_words == 0:
        return _fallback_uniform_duration(scenes, total_duration)

    for scene in scenes:
        scene_text_norm = normalize_text(scene.text_vi)
        scene_words = scene_text_norm.split()
        scene_str = " ".join(scene_words)

        if not scene_words or cursor >= len(srt_words):
            # Gán theo tỷ lệ từ
            _assign_proportional(scene, scenes, total_duration, total_scene_words)
            logger.warning(f"[SemanticSRT] Scene {scene.scene_id}: no words or cursor exhausted, proportional.")
            continue

        # Giới hạn cửa sổ tìm kiếm ±2000 từ quanh cursor
        SEARCH_WINDOW = 2000
        search_start = cursor
        search_end = min(len(srt_words), cursor + len(scene_words) + SEARCH_WINDOW)
        search_str = " ".join(srt_words[search_start:search_end])

        matcher = difflib.SequenceMatcher(None, scene_str, search_str)
        ratio = matcher.ratio()

        if ratio >= 0.5:
            # Tìm block chứa từ khớp đầu và cuối
            matching_blocks = matcher.get_matching_blocks()
            if matching_blocks:
                # Tìm vị trí từ đầu tiên khớp trong search window
                first_match = matching_blocks[0]
                # b = vị trí trong search_str
                first_word_pos = len(search_str[:first_match.b].split()) - 1
                first_word_pos = max(0, first_word_pos)
                abs_first = search_start + first_word_pos

                # Tìm vị trí cuối
                last_match = matching_blocks[-2] if len(matching_blocks) > 1 else first_match
                last_word_pos = len(search_str[:last_match.b + last_match.size].split())
                abs_last = min(search_start + last_word_pos, len(srt_words) - 1)

                # Map về timing
                start_block_idx = word_to_block[min(abs_first, len(word_to_block) - 1)]
                end_block_idx = word_to_block[min(abs_last, len(word_to_block) - 1)]

                scene.start_time = srt_blocks[start_block_idx].start
                scene.end_time = srt_blocks[end_block_idx].end
                scene.duration_sec = max(0.1, scene.end_time - scene.start_time)

                cursor = abs_last + 1
            else:
                _assign_proportional(scene, scenes, total_duration, total_scene_words)
                logger.warning(f"[SemanticSRT] Scene {scene.scene_id}: no matching blocks, proportional.")
        else:
            _assign_proportional(scene, scenes, total_duration, total_scene_words)
            logger.warning(
                f"[SemanticSRT] Scene {scene.scene_id}: ratio {ratio:.2f} < 0.5, proportional timing."
            )

    # Post-process: đảm bảo tăng dần, không chồng lấn, cảnh cuối = total_duration
    _fix_monotonic(scenes, total_duration)
    return scenes


def _assign_proportional(scene, all_scenes, total_duration, total_words):
    """Gán timing theo tỷ lệ số từ (nội suy)."""
    words = len(scene.text_vi.split())
    proportion = words / total_words if total_words > 0 else 1.0 / max(len(all_scenes), 1)
    duration = total_duration * proportion
    # Tìm end_time của cảnh trước
    idx = next((i for i, s in enumerate(all_scenes) if s.scene_id == scene.scene_id), 0)
    prev_end = all_scenes[idx - 1].end_time if idx > 0 and all_scenes[idx - 1].end_time > 0 else 0.0
    scene.start_time = prev_end
    scene.end_time = prev_end + duration
    scene.duration_sec = duration


def _fix_monotonic(scenes, total_duration):
    """Sửa timing: tăng dần, không chồng lấn, cảnh cuối kết thúc = total_duration."""
    if not scenes:
        return
    for i in range(1, len(scenes)):
        if scenes[i].start_time < scenes[i - 1].end_time:
            scenes[i].start_time = scenes[i - 1].end_time
        if scenes[i].end_time <= scenes[i].start_time:
            scenes[i].end_time = scenes[i].start_time + 1.0
        scenes[i].duration_sec = scenes[i].end_time - scenes[i].start_time

    # Cảnh cuối = total_duration
    if scenes:
        scenes[-1].end_time = total_duration
        scenes[-1].duration_sec = max(0.1, scenes[-1].end_time - scenes[-1].start_time)


# ============================================================
# MAP SEMANTIC SCENES (Phase C)
# ============================================================

def map_semantic_scenes_to_srt(scenes, srt_blocks, total_duration):
    """Map semantic scenes (list of Scene dataclass) sang timeline bằng text matching.

    Thuật toán:
    1. Chuẩn hoá text 2 phía: lowercase, bỏ dấu câu, NFKD bỏ dấu tiếng Việt
    2. Ghép SRT blocks thành 1 chuỗi từ, ghi (block_index, word_position)
    3. Từng cảnh: difflib.SequenceMatcher tìm đoạn khớp (tuyến tính, không quay lui)
    4. Ratio < 0.5 → gán theo tỷ lệ số từ + log warning
    5. Bảo đảm: start/end tăng dần, không chồng lấn, cảnh cuối = total_duration
    """
    import unicodedata
    import difflib

    if not scenes:
        return scenes

    # Nếu không có SRT blocks → fallback tỷ lệ từ
    if not srt_blocks:
        logger.info("[SemanticSRT] No SRT blocks, using word-count proportional timing.")
        return _fallback_uniform_duration(scenes, total_duration)

    def normalize_text(text):
        """Lowercase, bỏ dấu tiếng Việt, bỏ dấu câu."""
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text

    # Ghép SRT thành chuỗi từ liên tục + map vị trí từ → block index
    srt_words = []
    word_to_block = []
    for blk_idx, block in enumerate(srt_blocks):
        words = normalize_text(block.text).split()
        for w in words:
            srt_words.append(w)
            word_to_block.append(blk_idx)

    cursor = 0  # vị trí từ hiện tại (tuyến tính, không quay lui)

    total_scene_words = sum(len(s.text_vi.split()) for s in scenes)
    if total_scene_words == 0:
        return _fallback_uniform_duration(scenes, total_duration)

    for scene in scenes:
        scene_text_norm = normalize_text(scene.text_vi)
        scene_words = scene_text_norm.split()
        scene_str = " ".join(scene_words)

        if not scene_words or cursor >= len(srt_words):
            _assign_proportional_timing(scene, scenes, total_duration, total_scene_words)
            logger.warning(
                f"[SemanticSRT] Scene {scene.scene_id}: "
                f"no words or cursor exhausted, proportional."
            )
            continue

        # Giới hạn cửa sổ tìm kiếm ±2000 từ quanh cursor
        SEARCH_WINDOW = 2000
        search_start = cursor
        search_end = min(len(srt_words), cursor + len(scene_words) + SEARCH_WINDOW)
        search_str = " ".join(srt_words[search_start:search_end])

        matcher = difflib.SequenceMatcher(None, scene_str, search_str)
        ratio = matcher.ratio()

        if ratio >= 0.5:
            matching_blocks = matcher.get_matching_blocks()
            if matching_blocks:
                first_match = matching_blocks[0]
                first_word_pos = len(search_str[:first_match.b].split()) - 1
                first_word_pos = max(0, first_word_pos)
                abs_first = search_start + first_word_pos

                last_match = matching_blocks[-2] if len(matching_blocks) > 1 else first_match
                last_word_pos = len(search_str[:last_match.b + last_match.size].split())
                abs_last = min(search_start + last_word_pos, len(srt_words) - 1)

                start_block_idx = word_to_block[min(abs_first, len(word_to_block) - 1)]
                end_block_idx = word_to_block[min(abs_last, len(word_to_block) - 1)]

                scene.start_time = srt_blocks[start_block_idx].start
                scene.end_time = srt_blocks[end_block_idx].end
                scene.duration_sec = max(0.1, scene.end_time - scene.start_time)

                cursor = abs_last + 1
            else:
                _assign_proportional_timing(scene, scenes, total_duration, total_scene_words)
                logger.warning(
                    f"[SemanticSRT] Scene {scene.scene_id}: "
                    f"no matching blocks, proportional."
                )
        else:
            _assign_proportional_timing(scene, scenes, total_duration, total_scene_words)
            logger.warning(
                f"[SemanticSRT] Scene {scene.scene_id}: "
                f"ratio {ratio:.2f} < 0.5, proportional timing."
            )

    # Post-process: đảm bảo tăng dần, không chồng lấn, cảnh cuối = total_duration
    _fix_monotonic(scenes, total_duration)
    return scenes


def _assign_proportional_timing(scene, all_scenes, total_duration, total_words):
    """Gán timing theo tỷ lệ số từ (nội suy) khi text matching thất bại."""
    words = len(scene.text_vi.split())
    proportion = words / total_words if total_words > 0 else 1.0 / max(len(all_scenes), 1)
    duration = total_duration * proportion
    # Tìm end_time của cảnh trước
    idx = next((i for i, s in enumerate(all_scenes) if s.scene_id == scene.scene_id), 0)
    prev_end = 0.0
    if idx > 0 and all_scenes[idx - 1].end_time > 0:
        prev_end = all_scenes[idx - 1].end_time
    scene.start_time = prev_end
    scene.end_time = prev_end + duration
    scene.duration_sec = duration


def _format_srt_time(t: float) -> str:
    """Giây → định dạng SRT HH:MM:SS,mmm."""
    t = max(0.0, t)
    hh = int(t // 3600)
    mm = int((t % 3600) // 60)
    ss = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        ms -= 1000
        ss += 1
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def generate_srt_from_scenes(scenes: List[Scene], out_path: str) -> str:
    """M1: Tự tạo file SRT từ kịch bản đã cắt cảnh (audio đọc NGUYÊN VĂN kịch bản).

    Block phụ đề chia theo CÂU (không phải theo cảnh — cảnh 15s làm 1 block là
    quá dài để đọc); mỗi câu nhận timing tỷ lệ số từ BÊN TRONG cảnh của nó.
    Yêu cầu: scenes đã có start_time/end_time (gọi SAU bước map timeline).
    """
    import re as _re

    entries = []
    idx = 1
    for scene in scenes:
        text = " ".join((scene.text_vi or "").split())
        if not text:
            continue
        # Tách câu (giữ dấu kết câu); câu quá dài (>22 từ) cắt tiếp theo dấu phẩy
        sentences = [s.strip() for s in _re.split(r'(?<=[\.\!\?\…;])\s+', text) if s.strip()]
        chunks = []
        for s in sentences:
            words = s.split()
            if len(words) <= 22:
                chunks.append(s)
            else:
                parts = [p.strip() for p in _re.split(r'(?<=,)\s+', s) if p.strip()]
                buf = ""
                for p in parts:
                    if buf and len((buf + " " + p).split()) > 22:
                        chunks.append(buf)
                        buf = p
                    else:
                        buf = (buf + " " + p).strip()
                if buf:
                    chunks.append(buf)
        if not chunks:
            continue

        scene_dur = max(scene.duration_sec, 0.2)
        wcounts = [max(len(c.split()), 1) for c in chunks]
        total_w = sum(wcounts)
        cursor = scene.start_time
        for chunk, w in zip(chunks, wcounts):
            seg = scene_dur * w / total_w
            end = min(cursor + seg, scene.end_time)
            if end <= cursor:
                end = cursor + 0.2
            entries.append(f"{idx}\n{_format_srt_time(cursor)} --> {_format_srt_time(end)}\n{chunk}\n")
            idx += 1
            cursor = end

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries) + "\n")
    logger.info(f"[SRT-Gen] Đã tạo {idx - 1} block phụ đề từ kịch bản → {out_path}")
    return out_path
