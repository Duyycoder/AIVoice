import os
import json
import time
import re
from loguru import logger
from openai import OpenAI
from app.config import config

def parse_srt(srt_content: str) -> list[dict]:
    """Parses SRT file content into a list of segment dictionaries."""
    segments = []
    lines = srt_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if line is subtitle index
        if line.isdigit():
            idx = int(line)
            i += 1
            if i >= len(lines):
                break
            
            timestamp = lines[i].strip()
            i += 1
            
            text_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                text_lines.append(lines[i].strip())
                i += 1
            
            text = " ".join(text_lines)
            segments.append({
                "index": idx,
                "timestamp": timestamp,
                "text": text
            })
        else:
            i += 1
            
    return segments

def build_srt(segments: list[dict]) -> str:
    """Builds SRT file content from a list of segment dictionaries."""
    lines = []
    for seg in segments:
        lines.append(str(seg["index"]))
        lines.append(seg["timestamp"])
        lines.append(seg["text"])
        lines.append("")  # Empty line separator
    return "\n".join(lines)

def translate_srt(srt_path: str, output_path: str, source_lang: str, target_lang: str = "Vietnamese") -> str:
    """Translates an SRT file from source_lang to target_lang using local Gemini API in batches."""
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        srt_content = f.read()
        
    segments = parse_srt(srt_content)
    if not segments:
        logger.warning(f"No subtitle segments found to translate in {srt_path}")
        # Just copy input to output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        return output_path

    api_key = config.app.get("openai_api_key")
    if not api_key:
        logger.error("No API key configured for LLM. Skipping translation, copying original subtitles.")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        return output_path
        
    client = OpenAI(
        api_key=api_key,
        base_url=config.app.get("openai_base_url", "https://api.openai.com/v1")
    )
    model = config.app.get("openai_model", "gemini-2.0-flash")
    
    logger.info(f"Translating {len(segments)} segments from {source_lang} to {target_lang} using {model}...")
    
    batch_size = 40
    translated_segments = []
    
    for start_idx in range(0, len(segments), batch_size):
        batch = segments[start_idx:start_idx + batch_size]
        logger.info(f"Translating batch: segments {batch[0]['index']} to {batch[-1]['index']}")
        
        # Prepare lightweight JSON payload
        payload = [{"i": seg["index"], "t": seg["text"]} for seg in batch]
        payload_str = json.dumps(payload, ensure_ascii=False)
        
        prompt = f"""You are a professional subtitle translator. Translate the following subtitle segments from {source_lang} to {target_lang}.
Maintain the exact same JSON array structure, matching each translation back to its original index 'i'.
Ensure the translations are natural, contextual, and fit well as video subtitles (concise yet accurate).

Source JSON:
{payload_str}

Return ONLY the translated JSON array of objects, with keys 'i' and 't'. Do not include markdown code block wrappers (like ```json), introduction, or explanation.
Example response:
[
  {{"i": 1, "t": "Xin chào thế giới."}},
  {{"i": 2, "t": "Chúc một ngày tốt lành."}}
]
"""
        
        translated_batch_map = {}
        success = False
        
        # Retry up to 3 times
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048
                )
                content = response.choices[0].message.content.strip()
                
                # Strip markdown code blocks
                if content.startswith("```json"):
                    content = content.replace("```json", "", 1).strip()
                elif content.startswith("```"):
                    content = content.replace("```", "", 1).strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
                
                translated_list = json.loads(content)
                if isinstance(translated_list, list):
                    for item in translated_list:
                        if isinstance(item, dict) and "i" in item and "t" in item:
                            translated_batch_map[int(item["i"])] = item["t"]
                    success = True
                    break
                else:
                    logger.warning(f"Translation response is not a JSON list: {content[:100]}")
            except Exception as e:
                logger.warning(f"Translation attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    err_str = str(e).lower()
                    if "429" in err_str or "quota" in err_str or "rate" in err_str or "resource_exhausted" in err_str:
                        wait_seconds = 35.0  # Default fallback
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                        if match:
                            wait_seconds = float(match.group(1)) + 1.0  # Add 1s safety margin
                        logger.warning(f"Rate limit hit. Waiting {wait_seconds:.1f}s before retrying...")
                        time.sleep(wait_seconds)
                    else:
                        time.sleep(2.0)  # Wait 2 seconds for other failures before retry
                
        # Merge translations back
        for seg in batch:
            original_index = seg["index"]
            translated_text = translated_batch_map.get(original_index, seg["text"])
            translated_segments.append({
                "index": original_index,
                "timestamp": seg["timestamp"],
                "text": translated_text
            })
            
    # Write translated SRT
    translated_srt_content = build_srt(translated_segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated_srt_content)
        
    logger.info(f"Translated subtitle saved to: {output_path}")
    return output_path
