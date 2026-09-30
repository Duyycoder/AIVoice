# -*- coding: utf-8 -*-
"""Tách cảnh ngữ nghĩa: LLM chỉ đánh dấu ĐIỂM BẮT ĐẦU cảnh, code dựng + cân độ dài.

Bản cũ bắt LLM viết lại cả danh sách cảnh (paragraphs/summary/action...) cho cả
chương trong 1 lời gọi: đề bài ~2.900 token + trả lời ~3.850 token > ngữ cảnh 4096
của Ollama -> Ollama xoá bớt đầu ngữ cảnh (chính là truyện) khi đang viết -> cảnh
sau bịa, số đoạn vượt chương, JSON cụt; lại chờ 60s x 3 lần rồi rơi về md_parser.

Bây giờ:
- Chia chương thành phần ~SPLIT_CHUNK_WORDS từ; mỗi lời gọi chỉ trả mốc bắt đầu
  + địa điểm/thời gian/nhân vật/hành động ngắn -> đề + trả lời vừa ~3.200 token.
- Code bỏ mốc sai, gộp cảnh cùng địa điểm, gộp cảnh ngắn, ép tổng số cảnh
  (~30-60s/cảnh), cắt cảnh quá dài.
- LLM hỏng phần nào thì phần đó chia thuần bằng code (vẫn ra ~12 cảnh/chương,
  không đẩy 50-70 cảnh sang bước vẽ ảnh). Chỉ trả None khi kịch bản rỗng.
"""
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from loguru import logger

# Mỗi phần ~1.100 từ ≈ 1.800 token tiếng Việt (Qwen) + system ~400 + trả lời
# SPLIT_MAX_TOKENS -> dưới 4096 (num_ctx mặc định của Ollama).
SPLIT_CHUNK_WORDS = 1100
SPLIT_MAX_TOKENS = 1000
# Trần cho 1 lời gọi, chỉ để không treo vĩnh viễn khi Ollama chết. Không retry:
# retry cùng đề chỉ đốt thêm thời gian như lần đầu.
SPLIT_LLM_TIMEOUT_SEC = 600.0

# Độ dài cảnh mong muốn (giây, ước theo tỷ lệ số từ trên thời lượng audio).
SCENE_MIN_SEC = 20.0
SCENE_TARGET_SEC = 40.0
SCENE_MAX_SEC = 60.0


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
    min_scene_sec: float = SCENE_MIN_SEC,
    max_scene_sec: float = SCENE_MAX_SEC,
) -> Optional[List[SemanticScene]]:
    """Tách cảnh. Trả None chỉ khi không có đoạn nào (caller fallback md_parser)."""
    paragraphs = _extract_paragraphs(md_text)
    if not paragraphs:
        return None

    boundaries = _call_llm_boundaries(paragraphs, total_audio_duration)
    ranges = _boundaries_to_ranges(boundaries, len(paragraphs))
    words = [len(p.split()) for p in paragraphs]
    ranges = _normalize_ranges(ranges, words, total_audio_duration,
                               min_scene_sec, max_scene_sec)
    scenes = _build_scenes(ranges, paragraphs)
    logger.info(f"[SemanticSplit] {len(paragraphs)} đoạn → {len(scenes)} cảnh "
                f"(~{total_audio_duration / max(len(scenes), 1):.0f}s/cảnh)")
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

    # FIX 06/07 (khôi phục sau sự cố move): CHIA NHỎ đoạn quá dài thành cụm câu
    # (~35 từ/cụm). File .md ít xuống dòng (truyện convert: cả chương = 1-2 khối)
    # sẽ chỉ có 1-2 "đoạn" khổng lồ → LLM chỉ được ghép ĐOẠN LIỀN KỀ thành cảnh
    # nên không thể chia mịn hơn đơn vị đoạn → cả video chỉ 2 cảnh.
    units = []
    for p in merged:
        if len(p.split()) <= 60:
            units.append(p)
            continue
        sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?\…;])\s+', p) if s.strip()]
        buf = ""
        for s in sentences:
            candidate = (buf + " " + s).strip()
            if buf and len(candidate.split()) > 35:
                units.append(buf)
                buf = s
            else:
                buf = candidate
        if buf:
            units.append(buf)

    logger.info(f"[SemanticSplit] Kịch bản: {len(paragraphs)} đoạn gốc → "
                f"{len(units)} đơn vị chia cảnh ({sum(len(u.split()) for u in units)} từ)")
    return units


# ----------------------------------------------------------------------
# LLM: chỉ lấy mốc bắt đầu cảnh
# ----------------------------------------------------------------------

def _chunk_ranges(words: List[int], chunk_words: int) -> List[tuple]:
    """Chia chỉ số đoạn thành các khoảng [lo, hi) mỗi khoảng <= chunk_words từ."""
    chunks, lo, acc = [], 0, 0
    for i, w in enumerate(words):
        if acc and acc + w > chunk_words:
            chunks.append((lo, i))
            lo, acc = i, 0
        acc += w
    chunks.append((lo, len(words)))
    return chunks


def _build_split_prompt(first: int, last: int, part_sec: float, prev_location: str) -> str:
    want = max(1, round(part_sec / SCENE_TARGET_SEC))
    prev = (f"Cảnh ngay trước phần này diễn ra ở: {prev_location}. Nếu đoạn [{first}] "
            f"vẫn ở đó thì vẫn ghi start {first} với cùng địa điểm.\n" if prev_location else "")
    return (
        "Bạn chia cảnh cho video truyện kể. Người dùng gửi một phần chương truyện, "
        f"mỗi đoạn có số [{first}]..[{last}].\n"
        "Hãy chỉ ra các đoạn MỞ ĐẦU một cảnh mới. Chỉ mở cảnh mới khi đổi địa điểm, "
        "đổi thời gian, hoặc hành động chính thay đổi hẳn. Hội thoại liên tục ở cùng "
        "chỗ là CÙNG MỘT cảnh.\n"
        f"Phần này dài khoảng {part_sec:.0f} giây, cần khoảng {want} cảnh "
        f"(mỗi cảnh 30-50 giây). Cảnh đầu tiên luôn bắt đầu ở đoạn {first}. "
        f"Chỉ dùng số đoạn từ {first} đến {last}.\n"
        f"{prev}"
        # qwen2.5 hay tự chuyển sang tiếng Trung (杜大壮 thay cho Đỗ Đại Tráng) ->
        # tên không khớp danh sách nhân vật, cảnh mất identity.
        "Viết location/characters/action bằng TIẾNG VIỆT; tên nhân vật chép NGUYÊN VĂN "
        "như trong truyện, không dịch sang chữ Hán.\n"
        "Trả về JSON, không viết gì khác:\n"
        '{"scenes": [{"start": <số đoạn>, "location": "địa điểm ngắn", '
        '"time_of_day": "day|night|dawn|dusk", "characters": ["Tên nhân vật"], '
        '"action": "hành động chính, tối đa 10 từ"}]}'
    )


# Nhắc lại NGAY SAU truyện: đặt trong system prompt thì qwen2.5 vẫn trả chữ Hán
# (đo thật 01/10: 3/3 lần). Ví dụ tiếng Việt kéo câu trả lời về đúng ngôn ngữ.
SPLIT_USER_TAIL = (
    "\n\n---\nTrả lời JSON bằng TIẾNG VIỆT, tên nhân vật chép nguyên văn như trong "
    'truyện. Ví dụ: {"scenes": [{"start": 0, "location": "sân trước võ quán", '
    '"time_of_day": "day", "characters": ["Lý Mộ Sinh"], "action": "khiêng bàn ra hậu viện"}]}'
)


def _call_llm_boundaries(paragraphs: List[str], total_duration: float) -> List[dict]:
    """Mốc cảnh từ LLM theo từng phần. Phần nào LLM hỏng thì mọi đoạn của phần đó
    thành mốc (không metadata) để _normalize_ranges gộp bằng code."""
    words = [len(p.split()) for p in paragraphs]
    total_words = sum(words) or 1
    chunks = _chunk_ranges(words, SPLIT_CHUNK_WORDS)

    try:
        from app.services.llm import get_llm_client
        client, model = get_llm_client()
        client = client.with_options(
            timeout=httpx.Timeout(SPLIT_LLM_TIMEOUT_SEC, connect=10.0), max_retries=0)
    except Exception as e:
        logger.warning(f"[SemanticSplit] Không có LLM ({e}) — chia cảnh bằng code.")
        return [{"start": i} for i in range(len(paragraphs))]

    boundaries: List[dict] = []
    prev_location = ""
    for ci, (lo, hi) in enumerate(chunks):
        part_sec = sum(words[lo:hi]) / total_words * total_duration
        system_prompt = _build_split_prompt(lo, hi - 1, part_sec, prev_location)
        user_msg = "\n\n".join(f"[{i}] {paragraphs[i]}" for i in range(lo, hi)) + SPLIT_USER_TAIL
        got = None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_msg}],
                temperature=0.2,
                max_tokens=SPLIT_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = _parse_json_response(content)
            if parsed is None:
                _log_llm_error(f"chunk_{ci}_parse", content)
            else:
                got = []
                for b in parsed.get("scenes", []):
                    s = _as_int(b.get("start")) if isinstance(b, dict) else None
                    if s is not None and lo <= s < hi:
                        got.append(dict(b, start=s))
        except Exception as e:
            logger.error(f"[SemanticSplit] LLM error phần {ci + 1}/{len(chunks)}: {e}")
            _log_llm_error(f"chunk_{ci}", str(e))

        if not got:
            logger.warning(f"[SemanticSplit] Phần {ci + 1}/{len(chunks)} (đoạn {lo}-{hi - 1}): "
                           "LLM không dùng được — chia bằng code.")
            got = [{"start": i} for i in range(lo, hi)]
        boundaries.extend(got)
        prev_location = next((b.get("location", "") for b in reversed(got)
                              if b.get("location")), "")
    return boundaries


def _as_int(v) -> Optional[int]:
    """LLM hay trả số đoạn dạng "5" hoặc 5.0."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    try:
        f = float(str(v).strip().strip("[]"))
        return int(f) if f.is_integer() else None
    except (TypeError, ValueError):
        return None


def _parse_json_response(content: str) -> Optional[dict]:
    """Parse JSON response từ LLM, xử lý edge cases."""
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start == -1 or end == -1 or end < start:
            return None
        parsed = json.loads(content[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------------
# Code: mốc -> khoảng đoạn -> cân độ dài
# ----------------------------------------------------------------------

def _clean_meta(b: dict) -> dict:
    chars = b.get("characters") or []
    if not isinstance(chars, list):
        chars = [chars]
    return {
        "location": str(b.get("location") or "").strip(),
        "time_of_day": str(b.get("time_of_day") or "").strip(),
        "characters": [str(c).strip() for c in chars if str(c).strip()],
        "action": str(b.get("action") or "").strip(),
    }


def _boundaries_to_ranges(boundaries: List[dict], n: int) -> List[dict]:
    """Mốc (có thể trùng/lộn xộn/vượt chương) -> các khoảng [start, end) phủ kín 0..n-1."""
    meta_by_start = {}
    for b in boundaries:
        s = b.get("start")
        if isinstance(s, int) and 0 <= s < n and s not in meta_by_start:
            meta_by_start[s] = _clean_meta(b)
    meta_by_start.setdefault(0, _clean_meta({}))
    starts = sorted(meta_by_start)
    return [dict(start=s, end=(starts[k + 1] if k + 1 < len(starts) else n), **meta_by_start[s])
            for k, s in enumerate(starts)]


def _same_place(a: dict, b: dict) -> bool:
    la, lb = a["location"].lower(), b["location"].lower()
    return bool(la) and la == lb and (a["time_of_day"] or "") == (b["time_of_day"] or "")


def _merge_pair(ranges: List[dict], i: int) -> None:
    """Gộp ranges[i+1] vào ranges[i] (tại chỗ)."""
    a, b = ranges[i], ranges.pop(i + 1)
    a["end"] = b["end"]
    a["location"] = a["location"] or b["location"]
    a["time_of_day"] = a["time_of_day"] or b["time_of_day"]
    a["characters"] = list(dict.fromkeys(a["characters"] + b["characters"]))
    if b["action"] and b["action"] != a["action"]:
        a["action"] = f"{a['action']}; {b['action']}" if a["action"] else b["action"]


def _normalize_ranges(ranges: List[dict], words: List[int], total_duration: float,
                      min_sec: float = SCENE_MIN_SEC,
                      max_sec: float = SCENE_MAX_SEC) -> List[dict]:
    """Gộp cảnh cùng chỗ / quá ngắn, ép tổng số cảnh, cắt cảnh quá dài."""
    total_words = sum(words) or 1
    ranges = [dict(r) for r in ranges]

    def dur(r):
        return sum(words[r["start"]:r["end"]]) / total_words * total_duration

    # 1. Cảnh liền nhau cùng địa điểm + thời gian -> một cảnh (nếu không quá dài)
    i = 0
    while i + 1 < len(ranges):
        a, b = ranges[i], ranges[i + 1]
        if _same_place(a, b) and dur(a) + dur(b) <= max_sec:
            _merge_pair(ranges, i)
        else:
            i += 1

    def merge_cost(i):
        # Ưu tiên gộp cặp cùng chỗ, rồi cặp ngắn nhất
        a, b = ranges[i], ranges[i + 1]
        return (0 if _same_place(a, b) else 1, dur(a) + dur(b))

    # 2. Cảnh quá ngắn -> gộp vào hàng xóm rẻ hơn
    while len(ranges) > 1:
        shortest = min(range(len(ranges)), key=lambda k: dur(ranges[k]))
        if dur(ranges[shortest]) >= min_sec:
            break
        cands = [k for k in (shortest - 1, shortest) if 0 <= k < len(ranges) - 1]
        _merge_pair(ranges, min(cands, key=merge_cost))

    # 3. Trần số cảnh: trung bình không dưới ~30s/cảnh
    max_count = max(1, math.ceil(total_duration / 30.0))
    while len(ranges) > max_count:
        _merge_pair(ranges, min(range(len(ranges) - 1), key=merge_cost))

    # 4. Cảnh quá dài (>max_sec, >= 2 đoạn) -> cắt ở đoạn gần giữa số từ
    out = []
    stack = list(reversed(ranges))
    while stack:
        r = stack.pop()
        if dur(r) <= max_sec or r["end"] - r["start"] < 2:
            out.append(r)
            continue
        half, acc, cut = sum(words[r["start"]:r["end"]]) / 2, 0, r["start"] + 1
        for k in range(r["start"], r["end"] - 1):
            acc += words[k]
            cut = k + 1
            if acc >= half:
                break
        first = dict(r, end=cut)
        cont = r["action"] if not r["action"] or r["action"].endswith("(tiếp)") else r["action"] + " (tiếp)"
        second = dict(r, start=cut, action=cont)
        stack.extend([second, first])
    return out


def _build_scenes(ranges: List[dict], paragraphs: List[str]) -> List[SemanticScene]:
    scenes = []
    for idx, r in enumerate(ranges):
        indices = list(range(r["start"], r["end"]))
        scenes.append(SemanticScene(
            scene_index=idx,
            text_vi="\n\n".join(paragraphs[i] for i in indices),
            summary_vi=r["action"],
            location=r["location"],
            characters=list(r["characters"]),
            time_of_day=r["time_of_day"],
            action=r["action"],
            paragraph_indices=indices,
        ))
    return scenes


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
