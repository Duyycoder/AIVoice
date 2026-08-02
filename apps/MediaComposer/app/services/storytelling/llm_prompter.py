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

def _locked_style(context: StoryContext) -> str:
    """Style KHÓA của truyện — nguồn sự thật duy nhất cho art direction.

    Trước đây hàm này là một chuỗi chèn cứng
    ``"(highly detailed background, cinematic lighting, <model>)"`` — mâu thuẫn
    trực tiếp với file style của truyện (vd storyboard.txt yêu cầu
    "flat vector illustration, minimal shading"). Hai tuyên bố style đánh nhau
    trong cùng một prompt là lý do mỗi frame ra một kiểu.
    """
    try:
        style = (context.get_positive_prompt() or "").strip().strip(",")
    except Exception:
        style = ""
    return style or "(masterpiece, best quality:1.2)"


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
    
    genre = context.genre or "default"
    genre_rules = GENRE_SPECIFIC_RULES.get(genre, GENRE_SPECIFIC_RULES["default"])
    locked_style = _locked_style(context)

    return f"""You are an expert prompt engineer for cinematic Stable Diffusion image generation.
IMPORTANT: You are a TEXT-ONLY AI. DO NOT generate images. Your only job is to write a TEXT string (a prompt) that will be used by another system.

Given Vietnamese text from a novel scene, output a JSON object matching exactly this schema:
{{
  "image_prompt": "tag1, tag2, tag3",
  "action": "short English tags for the MAIN physical action of this scene",
  "characters": ["Name1", "Name2"],
  "primary_character": "Name1",
  "shot_type": "close|medium|wide",
  "background_prompt": "optional string - clean background description for separate BG rendering",
  "layout": [
    {{"name": "Name1", "anchor_x": 0.5, "anchor_y": 0.9, "scale": 0.8, "z": 0,
      "prompt": "short English pose and expression tags for this character"}}
  ]
}}

{genre_rules}

**SHOT TYPE RULES:**
- "close": dialogue, strong emotion, facial reactions, 1 character focus → camera tags like "upper body, portrait"
- "medium": 1-2 characters interacting, actions with hands/objects → "cowboy shot, medium shot"
- "wide": landscapes, establishing shots, crowds, travel, battles → "wide shot, establishing shot"
Pick the type that best serves THIS scene's storytelling. Roughly 30% close, 40% medium, 30% wide across a story.

**CRITICAL SD PROMPT RULES:**
1. **SIMPLE TAGS ONLY (ENGLISH):** The "image_prompt" MUST be strictly in ENGLISH. Use ONLY a simple, comma-separated list of keywords/tags (e.g., "mountain peak, cloudy sky, sunset"). NO full sentences, NO complex grammar.
2. **ENVIRONMENT FIRST:** Describe the concrete setting — place, objects, weather, time of day (e.g., "ancient temple courtyard, stone steps, pine trees, falling snow"). Characters are placed within this environment. NEVER use "simple background" or "white background".
3. **NEVER WRITE STYLE TAGS (CRITICAL):** The art style is FIXED by the system and already applied — it is: "{locked_style}". Your output must contain ONLY concrete, physical, describable content: subjects, clothing, objects, place, weather, time of day. FORBIDDEN in your output: "masterpiece", "best quality", "highres", "detailed", "intricate", "cinematic lighting", "depth of field", "anime", "digital art", "realistic", "3d", "8k", any model name, and any other tag about medium or render quality. Writing style tags makes every frame look different — that is the single worst failure mode.
4. **CHARACTER HANDLING:** If characters appear in the scene:
   a. In "characters"/"primary_character" fields use their EXACT NAME from the list below. But do NOT put names inside "image_prompt" (CLIP cannot read Vietnamese names — they waste tokens; identity is handled by IP-Adapter).
   b. Copy their visual `keywords` from the list below into the image_prompt.
   c. Add "solo" if only one character. Use "2girls" or "1boy 1girl" etc. for multiple.
   d. If the scene is purely landscape/establishing shot, use "no humans, scenery".
   e. Must clearly describe male characters as not effeminate. Minimize 'young'.
   Example: "1boy, male focus, black robe, handsome, wide shot, ancient marketplace"
5. **CAMERA:** Add ONE camera tag at the end (e.g., "wide shot"). No lighting/mood tags — those belong to the fixed style.
6. **CONTINUITY:** Read the "Director's Note" to understand the visual context and ensure the environment matches the story's progression.
7. **ACTION FIELD (CRITICAL):** The `action` field is what makes the scene READ as a moment instead of a portrait. Put the main physical action there as 2-4 concrete English verb tags — what the body is DOING, what the hands HOLD, which way it MOVES. Good: "swinging a sword downward, lunging forward, cape flying". Bad: "feeling sad", "being powerful", "standing". Never write "standing" or "looking at viewer" — those are the default and waste the field. If the scene is genuinely static, describe the specific posture instead ("kneeling with head bowed", "leaning against the doorframe"). The system applies attention weights to this field itself — do NOT add weight syntax yourself.
8. **LENGTH (CRITICAL — weak image model):** Keep "image_prompt" UNDER 35 words. The downstream image model is WEAK: it follows SHORT, CONCRETE prompts far better than long ones. Use only high-impact visual nouns (subject, setting, key objects, 1 camera tag). DROP vague adjectives, mood words, and redundant synonyms. Fewer, stronger tags beat many weak ones.
9. **STUDIO LAYOUT:** Include every listed scene character exactly once in `layout`. Its `prompt` contains ONLY that character's English pose, action, held object, and expression; never scenery, never appearance, never another character.

Known characters: {json.dumps(char_list, ensure_ascii=False)}
Story genre: {genre}"""

def _process_scene_with_retry(scene: Scene, system_prompt: str, context: StoryContext, retries: int = 1, director_note: str = "") -> bool:
    if scene.image_prompt:
        return True # Already processed
        
    user_prompt = f"Scene text:\n{scene.text_vi}"
    if director_note:
        user_prompt += f"\n\nDirector's Note (Visual Context):\n{director_note}"
    # Semantic scene metadata (Phase C) — attr động, scene_to_dict giữ qua resume
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
    
    from app.config import load_storytelling_config
    from app.services.storytelling.style_lock import build_locked_prompt

    locked_style = _locked_style(context)
    action_weight = float(load_storytelling_config().get("studio_action_weight", 1.35))

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
            raw_action = str(data.get("action", "") or "").strip()
            if raw_prompt:
                # Style luôn do hệ thống áp, hành động luôn được gắn trọng số ở
                # đầu prompt. LLM chỉ đóng góp phần nội dung cụ thể.
                scene.image_prompt = build_locked_prompt(
                    locked_style, raw_prompt,
                    action=raw_action, action_weight=action_weight)
            else:
                scene.image_prompt = ""
            scene._llm_action = raw_action

            scene.characters_in_scene = data.get("characters", [])
            scene.primary_character = data.get("primary_character") or ""

            # T5: cỡ cảnh — quyết định khung sinh ảnh (close=dọc mặt to)
            raw_shot = str(data.get("shot_type", "")).strip().lower()
            scene.shot_type = raw_shot if raw_shot in ("close", "medium", "wide") else "wide"
            
            # P3b: Layout LLM (tuỳ chọn)
            scene._llm_background_prompt = data.get("background_prompt")
            scene._llm_layout = data.get("layout")

            # SAFETY NET: prompt vượt 77 token đã có compel xử lý; chặn trần 75 từ
            # để tránh LLM viết lan man làm loãng composition.
            if scene.image_prompt:
                from app.services.storytelling.prompt_translator import PromptTranslator
                scene.image_prompt = PromptTranslator._clamp_prompt_words(scene.image_prompt, max_words=75)
                return True
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from LLM on attempt {attempt+1}. Content: {cleaned_content}")
            
    # Fallback if failed
    scene.image_prompt = build_locked_prompt(
        locked_style, "scenery, wide landscape, no humans")
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
