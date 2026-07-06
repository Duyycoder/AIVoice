# -*- coding: utf-8 -*-
"""Tách cảnh ngữ nghĩa bằng LLM — fallback None nếu LLM fail.

Gọi LLM để chia kịch bản thành các cảnh 8-15 giây theo ngữ cảnh.
Luôn có fallback: trả None để caller dùng md_parser cũ.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


@dataclass
class SemanticScene:
    scene_index: int
    text_vi: str          # Nguyên văn ghép các đoạn thuộc cảnh
    summary_vi: str
    location: str
    characters: List[str]  # Tên nhân vật
    time_of_day: str
    action: str
    paragraph_indices: List[int] = field(default_factory=list)


def split_scenes_semantic(
    md_text: str,
    total_audio_duration: float,
    min_scene_sec: float = 8.0,
    max_scene_sec: float = 15.0,
) -> Optional[List[SemanticScene]]:
    """Tách cảnh ngữ nghĩa. Trả None nếu LLM fail → caller fallback."""
    paragraphs = _extract_paragraphs(md_text)
    if not paragraphs:
        return None

    raw_scenes = _call_llm_split(paragraphs, total_audio_duration)
    if raw_scenes is None:
        return None

    if not _validate_scene_coverage(raw_scenes, len(paragraphs)):
        return None

    scenes = _build_scenes(raw_scenes, paragraphs)

    total_words = sum(len(p.split()) for p in paragraphs)
    scenes = _enforce_duration_bounds(
        scenes, total_words, total_audio_duration, min_scene_sec, max_scene_sec
    )
    return scenes


def _extract_paragraphs(md_text: str) -> List[str]:
    """Tái dùng logic lọc H1/quảng cáo từ md_parser.py."""
    lines = md_text.split('\n')
    paragraphs = []
    current_para = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue
        if line.startswith("# "):
            continue
        lower_line = line.lower()
        if "mời đọc" in lower_line or "bộ truyện về" in lower_line or "http" in lower_line:
            continue
        current_para.append(line)

    if current_para:
        paragraphs.append(" ".join(current_para))

    # Gộp paragraph ngắn
    merged = []
    for p in paragraphs:
        if len(p.split()) < 5 and merged:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def _call_llm_split(paragraphs: List[str],
                     total_duration: float) -> Optional[list]:
    """Gọi LLM để chia cảnh. Chunk 6000 từ nếu kịch bản dài."""
    try:
        from app.services.llm import get_llm_client
    except Exception:
        logger.warning("[SemanticSplit] LLM client not available")
        return None

    numbered = []
    for i, p in enumerate(paragraphs):
        numbered.append(f"[{i}] {p}")
    full_text = "\n\n".join(numbered)
    total_words = sum(len(p.split()) for p in paragraphs)

    system_prompt = (
        f"Bạn là chuyên gia tách cảnh cho video truyện kể.\n\n"
        f"Được cho kịch bản đánh số đoạn [0], [1], [2]... "
        f"và tổng thời lượng audio {total_duration:.0f} giây.\n\n"
        f"Hãy chia thành các cảnh sao cho:\n"
        f"- Mỗi cảnh ước tính kéo dài 8-15 giây "
        f"(dựa tỷ lệ số từ / tổng từ × {total_duration:.0f}s)\n"
        f"- Ranh giới cảnh = đổi địa điểm, đổi nhóm nhân vật, "
        f"đổi hành động chính, hoặc chuyển thời gian\n"
        f"- Mỗi cảnh là dãy đoạn LIÊN TIẾP, không bỏ sót, không chồng lấn\n\n"
        f"Output JSON (KHÔNG text ngoài JSON):\n"
        "{\n"
        '  "scenes": [\n'
        "    {\n"
        '      "paragraphs": [0, 1, 2],\n'
        '      "location": "mô tả ngắn địa điểm",\n'
        '      "characters": ["Tên nhân vật 1"],\n'
        '      "time_of_day": "day/night/dawn/dusk",\n'
        '      "action": "mô tả hành động chính",\n'
        '      "summary": "tóm tắt cảnh 1 câu"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    CHUNK_WORD_LIMIT = 6000
    if total_words <= CHUNK_WORD_LIMIT:
        chunks = [full_text]
    else:
        chunks = _split_into_chunks(numbered, CHUNK_WORD_LIMIT)

    all_scenes = []
    prev_context = ""
    for chunk_idx, chunk_text in enumerate(chunks):
        user_msg = chunk_text
        if prev_context:
            user_msg = (
                f"Ngữ cảnh nối (2 cảnh cuối chunk trước):\n"
                f"{prev_context}\n\n---\n\n{chunk_text}"
            )

        try:
            client, model = get_llm_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[SemanticSplit] LLM error chunk {chunk_idx}: {e}")
            _log_llm_error(f"chunk_{chunk_idx}", str(e))
            return None

        parsed = _parse_json_response(content)
        if parsed is None:
            _log_llm_error(f"chunk_{chunk_idx}_parse", content)
            return None

        scenes_data = parsed.get("scenes", [])
        if not scenes_data:
            return None

        all_scenes.extend(scenes_data)

        if len(scenes_data) >= 2:
            prev_context = json.dumps(scenes_data[-2:], ensure_ascii=False)
        elif scenes_data:
            prev_context = json.dumps(scenes_data[-1:], ensure_ascii=False)

    return all_scenes


def _split_into_chunks(numbered_paragraphs: List[str],
                       word_limit: int) -> List[str]:
    """Chia danh sách paragraph đã đánh số thành chunks theo giới hạn từ."""
    chunks = []
    current_chunk = []
    current_words = 0
    for para in numbered_paragraphs:
        words = len(para.split())
        if current_words + words > word_limit and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_words = 0
        current_chunk.append(para)
        current_words += words
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def _parse_json_response(content: str) -> Optional[dict]:
    """Parse JSON response từ LLM, xử lý edge cases."""
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start == -1 or end == -1 or end < start:
            return None
        return json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None


def _validate_scene_coverage(raw_scenes: list, n_paragraphs: int) -> bool:
    """Mọi paragraph index xuất hiện đúng 1 lần, liên tiếp, phủ kín 0..N-1."""
    if not raw_scenes:
        return False

    all_indices = []
    for scene in raw_scenes:
        paras = scene.get("paragraphs", [])
        if not paras:
            logger.warning("[SemanticSplit] Validation fail: cảnh không có paragraphs")
            return False
        all_indices.extend(paras)

    expected = list(range(n_paragraphs))
    if sorted(all_indices) != expected:
        logger.warning(
            f"[SemanticSplit] Validation fail: "
            f"indices {sorted(all_indices)} != expected {expected}"
        )
        return False

    for scene in raw_scenes:
        paras = scene["paragraphs"]
        if paras != list(range(min(paras), max(paras) + 1)):
            logger.warning(
                f"[SemanticSplit] Validation fail: "
                f"paragraphs not contiguous: {paras}"
            )
            return False

    prev_max = -1
    for scene in raw_scenes:
        paras = scene["paragraphs"]
        if min(paras) <= prev_max:
            logger.warning("[SemanticSplit] Validation fail: overlapping scenes")
            return False
        prev_max = max(paras)

    return True


def _build_scenes(raw_scenes: list,
                  paragraphs: List[str]) -> List[SemanticScene]:
    """Chuyển raw JSON scenes thành SemanticScene list."""
    scenes = []
    for idx, raw in enumerate(raw_scenes):
        para_indices = raw.get("paragraphs", [])
        text_parts = [paragraphs[i] for i in para_indices if i < len(paragraphs)]
        scenes.append(SemanticScene(
            scene_index=idx,
            text_vi="\n\n".join(text_parts),
            summary_vi=raw.get("summary", ""),
            location=raw.get("location", ""),
            characters=raw.get("characters", []),
            time_of_day=raw.get("time_of_day", ""),
            action=raw.get("action", ""),
            paragraph_indices=para_indices,
        ))
    return scenes


def _enforce_duration_bounds(
    scenes: List[SemanticScene],
    total_words: int,
    total_duration: float,
    min_sec: float,
    max_sec: float,
) -> List[SemanticScene]:
    """Post-rules: gộp cảnh quá ngắn, cắt cảnh quá dài."""
    if not scenes or total_words == 0:
        return scenes

    def est_duration(scene):
        words = len(scene.text_vi.split())
        return (words / total_words) * total_duration if total_words > 0 else 0

    # Gộp cảnh quá ngắn
    merged = []
    for scene in scenes:
        dur = est_duration(scene)
        if dur < min_sec and merged:
            prev = merged[-1]
            prev.text_vi += "\n\n" + scene.text_vi
            prev.paragraph_indices.extend(scene.paragraph_indices)
            prev.characters = list(set(prev.characters + scene.characters))
            if scene.action:
                prev.action += "; " + scene.action
        else:
            merged.append(scene)

    # Cắt cảnh quá dài (> max × 1.8)
    final = []
    for scene in merged:
        dur = est_duration(scene)
        if dur > max_sec * 1.8 and len(scene.paragraph_indices) >= 2:
            paras = scene.paragraph_indices
            mid = len(paras) // 2

            text_parts = scene.text_vi.split("\n\n")
            first_texts = text_parts[:mid]
            second_texts = text_parts[mid:]

            final.append(SemanticScene(
                scene_index=len(final),
                text_vi="\n\n".join(first_texts),
                summary_vi=scene.summary_vi,
                location=scene.location,
                characters=scene.characters,
                time_of_day=scene.time_of_day,
                action=scene.action,
                paragraph_indices=paras[:mid],
            ))
            final.append(SemanticScene(
                scene_index=len(final),
                text_vi="\n\n".join(second_texts),
                summary_vi=scene.summary_vi,
                location=scene.location,
                characters=scene.characters,
                time_of_day=scene.time_of_day,
                action=scene.action + " (tiếp)",
                paragraph_indices=paras[mid:],
            ))
        else:
            scene.scene_index = len(final)
            final.append(scene)

    return final


def _log_llm_error(tag: str, content: str):
    """Ghi mẫu response hỏng vào storage/logs/llm_errors/."""
    try:
        _mc_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        log_dir = os.path.join(_mc_root, "storage", "logs", "llm_errors")
        os.makedirs(log_dir, exist_ok=True)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"semantic_split_{tag}_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"[SemanticSplit] Logged error to {path}")
    except Exception:
        pass
