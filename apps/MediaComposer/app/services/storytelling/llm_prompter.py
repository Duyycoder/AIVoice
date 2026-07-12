import json
from typing import List, Callable, Optional
from loguru import logger
from app.services.llm import get_llm_client
from app.services.storytelling.models import Scene, StoryContext

def _call_llm(messages: List[dict], max_tokens: int = 800) -> str:
    try:
        client, model = get_llm_client()
    except Exception as e:
        logger.error(f"LLM chưa sẵn sàng (kiểm tra API Key trong Global Settings): {e}")
        return ""
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

GENRE_SPECIFIC_RULES = {
    "tien_hiep": """**GENRE-SPECIFIC VISUAL RULES (Tiên Hiệp / Cổ Trang):**
- Style: Traditional Chinese fantasy, martial arts (wuxia/xianxia), cultivation, and mythology.
- STRICTLY AVOID modern elements (no cars, no modern buildings, no electronics, no modern clothing/glasses).
- Environment: majestic misty mountains, floating islands, bamboo forests, ancient Chinese pavilions/temples, cultivation caves, celestial skies.
- Visual elements: spiritual energy aura, flying swords, traditional robes (Hanfu), long hair, glowing talismans.
- Use tags: "ancient Chinese robes, long hair, sword cultivation, misty mountains, traditional architecture".""",

    "ngon_tinh": """**GENRE-SPECIFIC VISUAL RULES (Ngôn Tình / Đô Thị):**
- Style: Modern realistic, emotional, romantic, and urban.
- Clothing: Contemporary fashion (suits, stylish shirts, modern dresses, casual attire).
- Environment: Modern city streets, cozy apartments, elegant offices, cafes, rainy streetscapes, parks.
- Focus: Rich facial expressions, eye contact, body language, romantic and warm atmosphere, soft lighting.
- Use tags: "modern fashion, emotional eyes, urban cafe, city streets, warm romantic lighting".""",

    "khoa_huyen": """**GENRE-SPECIFIC VISUAL RULES (Khoa Huyễn / Viễn Tưởng):**
- Style: Futuristic, high-tech, space exploration, and sci-fi.
- Clothing: High-tech spacesuits, futuristic armor, robotic implants, modern combat suits.
- Environment: Spaceships, high-tech control rooms, futuristic laboratories, neon-lit cyberpunk cities, alien planets.
- Visual elements: Holographic screens, glowing terminals, neon lights, outer space backgrounds, stars and nebulas.
- Use tags: "futuristic city, holographic displays, high-tech armor, neon lights, spaceship interior, advanced technology".""",

    "default": """**GENERAL VISUAL RULES:**
- Focus on high-quality cinematic framing, atmospheric lighting, and clear focus.
- Describe the setting, environment, weather, and lighting in detail to establish a strong visual context.
- Ensure characters are integrated naturally into the background."""
}

def _build_system_prompt(context: StoryContext) -> str:
    # Clean portrait-inducing tags from character lists sent to LLM
    char_list = []
    for c in context.characters:
        tags = [t.strip() for t in c.keywords_en.split(",") if t.strip()]
        blacklist = {"upper body", "looking at viewer", "close up", "portrait", "headshot", "face focus", "bust shot", "solo focus"}
        cleaned = [t for t in tags if t.lower() not in blacklist]
        char_list.append({
            "name": c.name,
            "description": c.description,
            "keywords": ", ".join(cleaned)
        })
    
    model_tag = "Anything V5:1.1"
    if context.checkpoint and "anything-v5" not in context.checkpoint.lower():
        model_tag = context.checkpoint.split("/")[-1].replace("-", " ").title()
        
    style_prefix = f"(highly detailed background, cinematic lighting, {model_tag})"
    
    genre = context.genre or "default"
    genre_rules = GENRE_SPECIFIC_RULES.get(genre, GENRE_SPECIFIC_RULES["default"])
    
    return f"""You are an expert prompt engineer for cinematic Stable Diffusion image generation.
IMPORTANT: You are a TEXT-ONLY AI. DO NOT generate images. Your only job is to write a TEXT string (a prompt) that will be used by another system.

Given Vietnamese text from a novel scene, output a JSON object matching exactly this schema:
{{
  "image_prompt": "tag1, tag2, tag3",
  "characters": ["Name1", "Name2"],
  "primary_character": "Name1",
  "shot_type": "close|medium|wide"
}}

{genre_rules}

**SHOT TYPE RULES:**
- "close": dialogue, strong emotion, facial reactions, 1 character focus → camera tags like "upper body, portrait"
- "medium": 1-2 characters interacting, actions with hands/objects → "cowboy shot, medium shot"
- "wide": landscapes, establishing shots, crowds, travel, battles → "wide shot, establishing shot"
Pick the type that best serves THIS scene's storytelling. Roughly 30% close, 40% medium, 30% wide across a story.

**CRITICAL SD PROMPT RULES:**
1. **SIMPLE TAGS ONLY (ENGLISH):** The "image_prompt" MUST be strictly in ENGLISH. Use ONLY a simple, comma-separated list of keywords/tags (e.g., "mountain peak, cloudy sky, sunset"). NO full sentences, NO complex grammar.
2. **ENVIRONMENT FIRST (CRITICAL):** Your PRIMARY focus is the background and environment. Start your tags by describing the scenery in high detail (e.g., "detailed background, majestic scenery, vast landscape, ancient temple, cinematic lighting, depth of field"). Characters are secondary elements placed within this environment. NEVER use "simple background" or "white background".
3. **STYLE PREFIX:** Always start the prompt with: "{style_prefix}, "
4. **CHARACTER HANDLING:** If characters appear in the scene:
   a. In "characters"/"primary_character" fields use their EXACT NAME from the list below. But do NOT put names inside "image_prompt" (CLIP cannot read Vietnamese names — they waste tokens; identity is handled by IP-Adapter).
   b. Copy their visual `keywords` from the list below into the image_prompt.
   c. Add "solo" if only one character. Use "2girls" or "1boy 1girl" etc. for multiple.
   d. If the scene is purely landscape/establishing shot, use "no humans, scenery".
   e. Must clearly describe male characters as not effeminate. Minimize 'young'.
   Example: "1boy, male focus, black robe, handsome, wide shot, ancient marketplace"
5. **CAMERA & LIGHTING:** Add ONE camera tag and ONE lighting tag at the end (e.g., "wide shot, cinematic lighting").
6. **ACTION & CONTINUITY:** Read the "Director's Note" to understand the visual context and ensure the environment matches the story's progression.
7. **ACTION WEIGHTING (CRITICAL):** Wrap the main action/pose of the scene in attention weight syntax `(action tags:1.3)`, placed right after the style prefix. Every distinct physical detail (what a character HOLDS, WHERE they are, weather) gets its own weighted tag, e.g. "(kneeling before the altar:1.3), (holding a glowing sword:1.35)". Without weights the model ignores these details.
8. **LENGTH (CRITICAL — weak image model):** Keep "image_prompt" UNDER 40 words. The downstream image model is WEAK: it follows SHORT, CONCRETE prompts far better than long ones. Use only high-impact visual nouns (subject, setting, key objects, 1 camera + 1 lighting tag). DROP vague adjectives, mood words, and redundant synonyms. Put subject + character appearance in the FIRST 20 words. Fewer, stronger tags beat many weak ones.

Known characters: {json.dumps(char_list, ensure_ascii=False)}
Story genre: {genre}"""

def _process_scene_with_retry(scene: Scene, system_prompt: str, context: StoryContext, retries: int = 1, director_note: str = "") -> bool:
    if scene.image_prompt:
        return True # Already processed
        
    user_prompt = f"Scene text:\n{scene.text_vi}"
    if director_note:
        user_prompt += f"\n\nDirector's Note (Visual Context):\n{director_note}"
    # Semantic scene metadata (Phase C) — attr động, mất sau resume nhưng chấp nhận được
    if hasattr(scene, '_semantic_meta') and scene._semantic_meta:
        meta = scene._semantic_meta
        if meta.get('location'):
            user_prompt += f"\nLocation: {meta['location']}"
        if meta.get('action'):
            user_prompt += f"\nAction: {meta['action']}"
        if meta.get('time_of_day'):
            user_prompt += f"\nTime of day: {meta['time_of_day']}"
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    model_tag = "Anything V5:1.1"
    if context.checkpoint and "anything-v5" not in context.checkpoint.lower():
        model_tag = context.checkpoint.split("/")[-1].replace("-", " ").title()
        
    style_prefix = f"(highly detailed background, cinematic lighting, {model_tag})"
    
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
                old_style = "(flat color, minimalist anime, clean lineart, Anything V5:1.1)"
                raw_prompt = raw_prompt.replace(old_style, "").strip(", ")
                if not raw_prompt.startswith("(highly detailed background"):
                    scene.image_prompt = f"{style_prefix}, {raw_prompt}"
                else:
                    scene.image_prompt = raw_prompt
            else:
                scene.image_prompt = ""
                
            scene.characters_in_scene = data.get("characters", [])
            scene.primary_character = data.get("primary_character") or ""

            # T5: cỡ cảnh — quyết định khung sinh ảnh (close=dọc mặt to)
            raw_shot = str(data.get("shot_type", "")).strip().lower()
            scene.shot_type = raw_shot if raw_shot in ("close", "medium", "wide") else "wide"

            # SAFETY NET: prompt vượt 77 token đã có compel xử lý; chặn trần 75 từ
            # để tránh LLM viết lan man làm loãng composition.
            if scene.image_prompt:
                from app.services.storytelling.prompt_translator import PromptTranslator
                scene.image_prompt = PromptTranslator._clamp_prompt_words(scene.image_prompt, max_words=75)
                return True
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON from LLM on attempt {attempt+1}. Content: {cleaned_content}")
            
    # Fallback if failed
    scene.image_prompt = f"{style_prefix}, scenery, majestic landscape, cinematic, depth of field"
    return False

def generate_storyboard_context(scenes: List[Scene], context: StoryContext) -> dict:
    """Bước 1: Story Director Pass. Gọi LLM để tóm tắt bối cảnh và hành động xuyên suốt."""
    if not scenes:
        return {}
        
    logger.info("Bước 1: Chạy Story Director để phân tích bối cảnh và hành động (Tiền xử lý)...")
    
    # Gộp toàn bộ văn bản
    full_script = ""
    for s in scenes:
        full_script += f"Scene {s.scene_id}: {s.text_vi}\n"
        
    genre = context.genre or "default"
    genre_rules = GENRE_SPECIFIC_RULES.get(genre, GENRE_SPECIFIC_RULES["default"])
    
    system_prompt = f"""You are a Storyboard Director for a cinematic movie.
IMPORTANT: You are a TEXT-ONLY AI. DO NOT generate images.
Read the following script and provide a brief 1-sentence visual direction (Director's Note) for each Scene.
Make sure the visual style matches the genre: {genre}.

{genre_rules}

Focus ONLY on: Who is in the frame, what are they doing, and where are they? Ensure continuity between scenes (if Scene 1 is in a courtyard, Scene 2 is likely still there unless stated otherwise).
Output ONLY a valid JSON dictionary mapping scene ID (as string) to the director's note.
Example: {{"0": "Dịch Phong is sitting in his wooden shop, looking bored.", "1": "Lạc Lan Tuyết walks into the shop, looking coldly at him."}}"""

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
    director_notes = generate_storyboard_context(scenes, context)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    total_scenes = len(scenes)
    for i in range(0, total_scenes, batch_size):
        batch = scenes[i:i + batch_size]
        logger.info(f"Processing LLM prompt for batch {i//batch_size + 1}, scenes {i} to {min(i+batch_size, total_scenes)-1} (Sequential/Local)")
        
        # Đặt max_workers=4 theo yêu cầu của user để test
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for scene in batch:
                note = director_notes.get(str(scene.scene_id), "")
                futures.append(executor.submit(_process_scene_with_retry, scene, system_prompt, context, 1, note))
                
            for future in as_completed(futures):
                pass  # Wait for all to finish
            
        if on_batch_complete:
            on_batch_complete(scenes)
            
    return scenes
