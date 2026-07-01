import codecs
from typing import List
from app.services.storytelling.models import Scene

def parse_md_to_scenes(
    md_path: str,
    total_audio_duration: float,
    scene_target_duration: float = 5.0
) -> List[Scene]:
    """
    Parse file .md → danh sách Scene chưa có SRT timing và image_prompt.
    Timing sẽ được điền bởi srt_mapper.py.
    """
    with codecs.open(md_path, "r", "utf-8-sig") as f:
        content = f.read()

    # 1. Loại bỏ header H1 và dòng quảng cáo, chia paragraph
    lines = content.split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue
        
        # Bỏ qua H1
        if line.startswith("# "):
            continue
            
        # Loại bỏ quảng cáo
        lower_line = line.lower()
        if "mời đọc" in lower_line or "bộ truyện về" in lower_line or "http" in lower_line:
            continue
            
        current_para.append(line)
        
    if current_para:
        paragraphs.append(" ".join(current_para))
        
    # Lọc paragraph ngắn và gộp
    merged_paragraphs = []
    for p in paragraphs:
        word_count = len(p.split())
        if word_count < 5 and merged_paragraphs:
            merged_paragraphs[-1] += " " + p
        else:
            merged_paragraphs.append(p)
            
    import re
    split_paragraphs = []
    for p in merged_paragraphs:
        if len(p.split()) > 20:
            sentences = re.split(r'(?<=[.!?])\s+', p)
            split_paragraphs.extend([s.strip() for s in sentences if s.strip()])
        else:
            split_paragraphs.append(p)
    merged_paragraphs = split_paragraphs

    # Tự động điều chỉnh scene_target_duration nếu file quá ngắn
    if len(merged_paragraphs) < 5:
        scene_target_duration = max(2.0, total_audio_duration / max(1, len(merged_paragraphs)))
        
    total_words = sum(len(p.split()) for p in merged_paragraphs)
    if total_words == 0:
        return []
        
    # Nhóm paragraph thành scenes
    scenes = []
    current_scene_text = []
    current_scene_words = 0
    current_scene_est_duration = 0.0
    
    scene_id = 0
    
    for p in merged_paragraphs:
        words = len(p.split())
        est_duration = (words / total_words) * total_audio_duration
        
        current_scene_text.append(p)
        current_scene_words += words
        current_scene_est_duration += est_duration
        
        if current_scene_est_duration >= scene_target_duration:
            scenes.append(Scene(
                scene_id=scene_id,
                text_vi="\n\n".join(current_scene_text),
                word_count=current_scene_words,
                start_time=0.0,
                end_time=0.0,
                duration_sec=0.0,
                image_prompt="",
                characters_in_scene=[],
                primary_character="",
                fallback_level=0,
                accepted_seed=-1,
                frame_path=""
            ))
            scene_id += 1
            current_scene_text = []
            current_scene_words = 0
            current_scene_est_duration = 0.0
            
    # Xử lý đoạn cuối còn sót lại
    if current_scene_text:
        if scenes:
            scenes[-1].text_vi += "\n\n" + "\n\n".join(current_scene_text)
            scenes[-1].word_count += current_scene_words
        else:
            scenes.append(Scene(
                scene_id=scene_id,
                text_vi="\n\n".join(current_scene_text),
                word_count=current_scene_words,
                start_time=0.0,
                end_time=0.0,
                duration_sec=0.0,
                image_prompt="",
                characters_in_scene=[],
                primary_character="",
                fallback_level=0,
                accepted_seed=-1,
                frame_path=""
            ))

    return scenes
