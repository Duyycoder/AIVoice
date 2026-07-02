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
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return ""

def _build_system_prompt(context: StoryContext) -> str:
    char_list = [{"name": c.name, "description": c.description, "keywords": c.keywords_en} for c in context.characters]
    
    return f"""You are an expert anime storyboard artist and prompt engineer for xianxia/cultivation novels.
IMPORTANT: You are a TEXT-ONLY AI. DO NOT generate images. Your only job is to write a TEXT string (a prompt) that will be used by another system.

Given Vietnamese text from a cultivation novel scene, output a JSON object matching exactly this schema:
{{
  "image_prompt": "tag1, tag2, tag3",
  "characters": ["Name1", "Name2"],
  "primary_character": "Name1"
}}

**CRITICAL SD PROMPT RULES:**
1. **ENGLISH ONLY TAGS:** The "image_prompt" MUST be strictly in ENGLISH. NO Vietnamese allowed! Use a comma-separated list of tags (e.g., "1boy, standing, sword"). DO NOT write full sentences.
2. **STYLE PREFIX:** Always start the prompt with: "(flat color, minimalist anime, clean lineart, Anything V5:1.1), "
3. **STRICT CHARACTER APPEARANCE (CRITICAL):** If a `primary_character` is in the scene, you MUST EXACTLY COPY ALL tags from their `keywords` in the JSON below into your prompt. If the character is male, you may gently add: "mature male, handsome, tall, well-built" but DO NOT make them overly muscular like a bodybuilder. Stick strictly to their context window.
4. **SINGLE CHARACTER FOCUS:** Use the tag "solo" if a character is present. DO NOT use "2boys", "2girls", etc. Use diverse camera angles (e.g., "wide shot", "cowboy shot", "medium shot", "from below", "over-the-shoulder") based on the Director's Note.
5. **ESTABLISHING SHOTS & SCENERY (CRITICAL):** The storyboard needs MORE background and landscape shots to feel like a real movie! If a scene introduces a new location, or focuses heavily on the atmosphere/building/sky, you MUST use "scenery, no humans, establishing shot, detailed environment" and COMPLETELY EXCLUDE all character tags. We do NOT want characters in every single frame. NEVER use "simple background". ALWAYS describe a specific, detailed location with cinematic lighting and depth of field.
6. **ACTION & CONTINUITY:** You MUST read the "Director's Note" to understand the flow. Extract the EXACT action from the text (e.g., "drinking tea", "fighting") and ensure the background aligns with the Director's Note for continuity.

Known characters: {json.dumps(char_list, ensure_ascii=False)}
Story genre: {context.genre}"""

def _process_scene_with_retry(scene: Scene, system_prompt: str, retries: int = 1, director_note: str = "") -> bool:
    if scene.image_prompt:
        return True # Already processed
        
    user_prompt = f"Scene text:\n{scene.text_vi}"
    if director_note:
        user_prompt += f"\n\nDirector's Note (Visual Context):\n{director_note}"
        
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
            # Extract JSON block safely
            start_idx = cleaned_content.find('{')
            end_idx = cleaned_content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                cleaned_content = cleaned_content[start_idx:end_idx+1]
            else:
                raise json.JSONDecodeError("No JSON object could be decoded", cleaned_content, 0)
            
            data = json.loads(cleaned_content)
            if isinstance(data, list):
                data = data[0] if len(data) > 0 else {}
            if not isinstance(data, dict):
                continue
            raw_prompt = data.get("image_prompt", "").strip()
            if raw_prompt:
                style_prefix = "(flat color, minimalist anime, clean lineart, Anything V5:1.1)"
                old_style = "(anime style, 2D flat illustration, cel shaded, clean lineart, Anything V5:1.1)"
                raw_prompt = raw_prompt.replace(old_style, "").strip(", ")
                if not raw_prompt.startswith("(flat color") and not raw_prompt.startswith("flat color"):
                    scene.image_prompt = f"{style_prefix}, {raw_prompt}"
                else:
                    scene.image_prompt = raw_prompt
            else:
                scene.image_prompt = ""
                
            scene.characters_in_scene = data.get("characters", [])
            scene.primary_character = data.get("primary_character") or ""
            if scene.image_prompt:
                return True
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON from LLM on attempt {attempt+1}. Content: {cleaned_content}")
            
    # Fallback if failed
    scene.image_prompt = "(flat color, minimalist anime, clean lineart, Anything V5:1.1), 1boy, traditional chinese clothing, simple background"
    return False

def generate_storyboard_context(scenes: List[Scene]) -> dict:
    """Bước 1: Story Director Pass. Gọi LLM để tóm tắt bối cảnh và hành động xuyên suốt."""
    if not scenes:
        return {}
        
    logger.info("Bước 1: Chạy Story Director để phân tích bối cảnh và hành động (Tiền xử lý)...")
    
    # Gộp toàn bộ văn bản
    full_script = ""
    for s in scenes:
        full_script += f"Scene {s.scene_id}: {s.text_vi}\n"
        
    system_prompt = """You are a Storyboard Director for a cinematic movie.
IMPORTANT: You are a TEXT-ONLY AI. DO NOT generate images.
Read the following script and provide a brief 1-sentence visual direction (Director's Note) for each Scene.
Focus ONLY on: Who is in the frame, what are they doing, and where are they? Ensure continuity between scenes (if Scene 1 is in a courtyard, Scene 2 is likely still there unless stated otherwise).
Output ONLY a valid JSON dictionary mapping scene ID (as string) to the director's note.
Example: {"0": "Dịch Phong is sitting in his wooden shop, looking bored.", "1": "Lạc Lan Tuyết walks into the shop, looking coldly at him."}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_script}
    ]
    
    content = _call_llm(messages, max_tokens=3000)
    if not content:
        logger.warning("Story Director không trả về kết quả.")
        return {}
        
    try:
        cleaned = content.strip()
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            cleaned = cleaned[start_idx:end_idx+1]
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Lỗi parse JSON từ Story Director: {e}")
        return {}

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
    
    # Pre-processing: Generate Director's Notes for all scenes
    director_notes = generate_storyboard_context(scenes)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    total_scenes = len(scenes)
    for i in range(0, total_scenes, batch_size):
        batch = scenes[i:i + batch_size]
        logger.info(f"Processing LLM prompt for batch {i//batch_size + 1}, scenes {i} to {min(i+batch_size, total_scenes)-1} (Parallel)")
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for scene in batch:
                note = director_notes.get(str(scene.scene_id), "")
                futures.append(executor.submit(_process_scene_with_retry, scene, system_prompt, 1, note))
                
            for future in as_completed(futures):
                pass  # Wait for all to finish
            
        if on_batch_complete:
            on_batch_complete(scenes)
            
    return scenes
