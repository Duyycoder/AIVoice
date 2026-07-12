import json
import re
from loguru import logger
from openai import OpenAI
from app.config import config

def extract_search_terms(subtitle_text: str, amount: int = 5) -> list[str]:
    """Uses LLM to extract visual search keywords from the transcribed subtitle text."""
    if not subtitle_text.strip():
        return []
        
    api_key = config.app.get("openai_api_key")
    if not api_key:
        logger.warning("No openai_api_key provided, returning fallback terms.")
        # Fallback to simple logic: grab first few words if no LLM
        words = [w for w in re.split(r'\W+', subtitle_text) if len(w) > 3]
        return words[:amount]

    client = OpenAI(
        api_key=api_key,
        base_url=config.app.get("openai_base_url", "https://api.openai.com/v1")
    )
    model = config.app.get("openai_model", "gpt-4o-mini")

    prompt = f"""
    Here is the transcription of a voiceover:
    "{subtitle_text}"

    Please extract or infer exactly {amount} search terms (keywords) that would be suitable for searching stock videos (Pexels, Pixabay) to match the context of this voiceover.
    Return ONLY a valid JSON array of strings, with no other text, no markdown block.
    Example: ["nature", "city timeline", "happy people", "sunset"]
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        # Clean up possible markdown wrappers
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        terms = json.loads(content)
        if isinstance(terms, list):
            return terms
        return []
    except Exception as e:
        logger.error(f"Failed to extract search terms: {e}. Falling back to offline keyword extraction.")
        words = [w for w in re.split(r'\W+', subtitle_text) if len(w) > 3]
        return words[:amount] if words else ["nature"]

class LLMNotConfiguredError(Exception):
    """Ngoại lệ khi chưa cấu hình API Key."""
    pass

def get_llm_client() -> tuple[OpenAI, str]:
    """Trả về (OpenAI client, model_name) dựa trên config hiện tại."""
    api_key = config.app.get("llm_api_key") or config.app.get("openai_api_key", "")
    base_url = config.app.get("llm_base_url") or config.app.get("openai_base_url", "https://api.openai.com/v1")
    model = config.app.get("llm_model") or config.app.get("openai_model", "gpt-4o-mini")
    if not api_key:
        raise LLMNotConfiguredError("Chưa cấu hình API Key. Vào Global Settings để thiết lập.")
    
    import httpx
    # Đặt timeout 60s để tránh treo tiến trình khi dùng Local LLM bị nghẽn
    return OpenAI(api_key=api_key, base_url=base_url, timeout=httpx.Timeout(60.0)), model


def _is_ollama_endpoint() -> bool:
    base_url = (config.app.get("llm_base_url") or config.app.get("openai_base_url", "")).lower()
    return "11434" in base_url or "ollama" in base_url


def unload_local_llm() -> None:
    """Nếu LLM hiện tại là Ollama (local), yêu cầu Ollama GIẢI PHÓNG model khỏi VRAM
    ngay lập tức (keep_alive=0).

    Gọi ở ranh giới pha LLM -> Stable Diffusion: Ollama qwen giữ ~3-5GB VRAM và
    (mặc định) neo 5 phút sau mỗi lần gọi, va chạm với SD trên GPU 6GB gây OOM.
    Best-effort: lỗi thì bỏ qua (vd đang dùng Gemini local, Ollama đã tắt...).
    """
    if not _is_ollama_endpoint():
        return
    base_url = config.app.get("llm_base_url") or config.app.get("openai_base_url", "")
    model = config.app.get("llm_model") or config.app.get("openai_model", "")
    try:
        import requests
        # base_url thường là ".../v1" — API native của Ollama nằm ở gốc host
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        requests.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10)
        logger.info(f"[LLM] Đã yêu cầu Ollama giải phóng model '{model}' khỏi VRAM (keep_alive=0).")
    except Exception as e:
        logger.warning(f"[LLM] Không unload được Ollama (bỏ qua): {e}")
