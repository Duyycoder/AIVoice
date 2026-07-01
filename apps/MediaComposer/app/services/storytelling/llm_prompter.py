import json
from typing import List, Callable, Optional
from loguru import logger
from app.services.llm import get_llm_client
from app.services.storytelling.models import Scene, StoryContext

def _call_llm(messages: List[dict], max_tokens: int = 500) -> str:
    client, model = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return ""

def _build_system_prompt(context: StoryContext) -> str:
    char_list = [{"name": c.name, "slug": c.slug, "description": c.description} for c in context.characters]
    
    return f"""You are an expert anime storyboard artist and prompt engineer for xianxia/cultivation novels.
Given Vietnamese text from a cultivation novel scene, output a JSON with:
1. "image_prompt": Stable Diffusion prompt in English (max 100 words)
2. "characters": list of character names appearing in this scene (from the provided list)
3. "primary_character": the most visually prominent character (or null if no character)

SD Prompt Rules:
- Style suffix (always append): "anime style, 2D flat illustration, cel shaded, clean lineart, Anything V5"
- Always specify shot type: close-up / medium shot / wide shot / aerial shot
- Always specify lighting: warm afternoon light / dramatic backlight / cool moonlight / etc.
- Always specify mood: tense / serene / shocked / determined / melancholic / etc.
- For xianxia settings use: traditional Chinese courtyard / ancient pagoda / mountain cliff / bamboo forest / cultivation chamber
- Character action must be explicit: "standing with arms crossed" not just "standing"
- Max 1 special effect (lightning / ki aura / spiritual energy) per prompt

Known characters: {json.dumps(char_list, ensure_ascii=False)}
Story genre: {context.genre}"""

def _process_scene_with_retry(scene: Scene, system_prompt: str, retries: int = 1) -> bool:
    if scene.image_prompt:
        return True # Already processed
        
    user_prompt = f"Scene text:\n{scene.text_vi}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(retries + 1):
        content = _call_llm(messages)
        if not content:
            continue
            
        try:
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            
            data = json.loads(cleaned_content)
            scene.image_prompt = data.get("image_prompt", "")
            scene.characters_in_scene = data.get("characters", [])
            scene.primary_character = data.get("primary_character") or ""
            if scene.image_prompt:
                return True
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from LLM on attempt {attempt+1}")
            
    # Fallback if failed
    scene.image_prompt = "1boy, traditional chinese clothing, anime style, 2D flat illustration, cel shaded, clean lineart, Anything V5"
    return False

def generate_prompts_batch(
    scenes: List[Scene],
    context: StoryContext,
    batch_size: int = 8,
    on_batch_complete: Optional[Callable[[List[Scene]], None]] = None
) -> List[Scene]:
    """
    Gọi LLM theo batch để giảm overhead.
    """
    system_prompt = _build_system_prompt(context)
    
    total_scenes = len(scenes)
    for i in range(0, total_scenes, batch_size):
        batch = scenes[i:i + batch_size]
        logger.info(f"Processing LLM prompt for batch {i//batch_size + 1}, scenes {i} to {min(i+batch_size, total_scenes)-1}")
        
        for scene in batch:
            _process_scene_with_retry(scene, system_prompt)
            
        if on_batch_complete:
            on_batch_complete(scenes)
            
    return scenes
