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
