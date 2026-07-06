import os
import json
import re
from typing import List, Tuple, Dict, Any
from loguru import logger

from app.services.llm import get_llm_client
from app.services.storytelling.context_manager import ContextManager
from app.services.storytelling.models import Character

# Absolute path to presets directory
_PRESETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "resource", "image_presets"))


TRANSLATE_SYSTEM_PROMPT = """You are an expert Stable Diffusion prompt engineer.
You receive:
1. A scene description.
2. A style specification (Positive and Negative).
3. A list of known characters with their visual keywords.

Your task: Combine ALL information into a single, highly optimized SD prompt in English.

RULES (STRICT):
1. Output ONLY a JSON with "positive" and "negative" fields.
2. Use comma-separated tags format. NO full sentences.
3. **Structure of positive prompt (VERY IMPORTANT):**
   [Style Tags from input], [(Main Action & Pose:1.3)], [Character Visual Keywords (e.g. "1boy, black robe, silver hair")], [Background & Setting], [Camera & Lighting]
   ⚠️ The main action and character appearance keywords MUST both appear within the FIRST 40 words.
4. **ACTION WEIGHTING (CRITICAL):** Wrap the main action/pose of the scene in attention weight syntax `(action tags:1.3)`. Every distinct physical detail the user asked for (what the character HOLDS, WHERE they stand, weather) must appear as its own weighted tag. Example: user says "đứng giữa phố đêm mưa, tay phải cầm ô" → "(standing in middle of street:1.3), (holding umbrella in right hand:1.35), (heavy rain:1.2)". Without weights the model ignores these details.
5. **LENGTH:** Keep the positive prompt UNDER 60 words. Prompts are encoded without truncation, but shorter prompts follow composition better. If trimming, cut camera/lighting first. Negative prompt under 50 words.
6. **STYLE:** Use the exact Style Positive/Negative provided. DO NOT add conflicting style tags (e.g. if the style is "watercolor", do not add "anime" or "photorealistic").
7. **CHARACTERS:**
   - Do NOT put character names in the prompt (CLIP cannot read Vietnamese names — they waste tokens). Identity is handled separately by IP-Adapter. Instead, copy their visual `keywords`.
   - ⚠️ CRITICAL: Anime models have a severe bias towards drawing females. If a character is male, you MUST emphasize male tags heavily (e.g. "1boy, male focus, mature man"). Do not use effeminate keywords.

Example output:
{
  "positive": "(masterpiece, best quality:1.2), flat color anime, (standing in middle of night street:1.3), (holding umbrella in right hand:1.35), (heavy rain:1.2), 1boy, male focus, long silver hair, red eyes, dark robes, night city background, street lights, cinematic lighting",
  "negative": "(worst quality:2), (low quality:2), female, girl, 1girl, blurry, 3d"
}"""

PRESET_GEN_SYSTEM_PROMPT = """You are an expert Stable Diffusion prompt engineer.
The user will describe an art style they want. Generate a reusable style preset.

Output ONLY a JSON with:
- "positive": Comma-separated style tags for SD (quality, art style, lighting, mood)
- "negative": Comma-separated negative tags to avoid

Keep each under 200 words. Focus on STYLE only, not content/subject. No markdown wrappers.
Example: {"positive": "watercolor painting, soft edges, traditional media, delicate colors", "negative": "photograph, 3d render, digital art, sharp edges"}"""


class PromptTranslator:
    def __init__(self, ctx_mgr: ContextManager):
        self.ctx_mgr = ctx_mgr
        self.characters = ctx_mgr.list_characters()
        os.makedirs(_PRESETS_DIR, exist_ok=True)

    def _call_llm_json(self, messages: List[dict], max_tokens: int = 800) -> dict:
        try:
            client, model = get_llm_client()
        except Exception as e:
            # LLMNotConfiguredError...: không crash cả trang, trả rỗng để dùng fallback prompt
            logger.error(f"LLM chưa sẵn sàng: {e}")
            return {}
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            # Clean markdown
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).strip()
            if content.endswith("```"):
                content = content[:-3].strip()
                
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error calling LLM for prompt translation: {e}")
            return {}

    def match_characters_in_text(self, text: str) -> List[Character]:
        """
        Fuzzy-match character names from Context Window in the user description.
        Returns matched Character objects.
        """
        if not self.characters or not text:
            return []
            
        import unicodedata
        def normalize(s):
            return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower().replace(' ', '').replace('_', '')
            
        text_norm = normalize(text)
        matched = []
        for char in self.characters:
            if normalize(char.name) in text_norm or normalize(char.slug) in text_norm:
                matched.append(char)
                
        # Remove duplicates while preserving order
        unique_matched = []
        for m in matched:
            if m not in unique_matched:
                unique_matched.append(m)
                
        return unique_matched

    @staticmethod
    def list_available_presets() -> List[str]:
        if not os.path.exists(_PRESETS_DIR):
            return []
        presets = []
        for f in os.listdir(_PRESETS_DIR):
            if f.endswith(".txt"):
                presets.append(f[:-4])
        return sorted(presets)

    @staticmethod
    def load_style_preset(preset_name: str) -> Tuple[str, str]:
        if not preset_name:
            return "", ""
            
        file_path = os.path.join(_PRESETS_DIR, f"{preset_name}.txt")
        if not os.path.exists(file_path):
            return "", ""
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            parts = content.split("---")
            pos = parts[0].strip()
            neg = parts[1].strip() if len(parts) > 1 else ""
            return pos, neg
        except Exception as e:
            logger.warning(f"Error loading preset {preset_name}: {e}")
            return "", ""

    def generate_custom_preset(self, style_description: str, preset_name: str) -> bool:
        """
        Uses LLM to generate a style preset from description and saves it to a file.
        Returns True if successful.
        """
        messages = [
            {"role": "system", "content": PRESET_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Style description: {style_description}"}
        ]
        
        result = self._call_llm_json(messages, max_tokens=500)
        
        pos = result.get("positive", "").strip()
        neg = result.get("negative", "").strip()
        
        if pos:
            try:
                # Ensure filename is safe
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', preset_name)
                if not safe_name:
                    safe_name = "custom_preset"
                
                file_path = os.path.join(_PRESETS_DIR, f"{safe_name}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"{pos}\n---\n{neg}")
                return True
            except Exception as e:
                logger.error(f"Error saving custom preset: {e}")
                return False
        return False

    def clean_portrait_tags(self, keywords_str: str) -> str:
        """Loại bỏ các tag quy định góc chụp chân dung để tránh đè bẹp bối cảnh rộng."""
        if not keywords_str:
            return ""
        tags = [t.strip() for t in keywords_str.split(",") if t.strip()]
        blacklist = {
            "upper body", "looking at viewer", "close up", "portrait", 
            "headshot", "face focus", "bust shot", "solo focus", "face shot"
        }
        cleaned = [t for t in tags if t.lower() not in blacklist]
        return ", ".join(cleaned)

    def adapt_style_to_model(self, pos: str, neg: str, checkpoint: str) -> Tuple[str, str]:
        """Tự động thích ứng style theo mô hình đang chạy (lọc tag model cũ, lọc negative block)."""
        if not checkpoint:
            return pos, neg
            
        ckpt_lower = checkpoint.lower()
        
        # 1. Xử lý Positive: Xóa 'Anything V5:1.1' nếu dùng model khác
        if "anything-v5" not in ckpt_lower:
            pos = re.sub(r',\s*Anything V5(:1\.1)?', '', pos, flags=re.IGNORECASE)
            pos = re.sub(r'Anything V5(:1\.1)?\s*,?', '', pos, flags=re.IGNORECASE)
            # Thêm tên model đang chạy nếu cần định hình rõ hơn cho CLIP
            model_short = checkpoint.split("/")[-1].replace("-", " ").title()
            if model_short:
                pos = f"({model_short}), {pos}".strip(", ")
            
        # 2. Xử lý Negative: Nếu là model realistic/dreamshaper, loại bỏ cấm 'realistic/photorealistic' ở negative
        is_realistic = "realistic" in ckpt_lower or "cyberrealistic" in ckpt_lower or "dreamshaper" in ckpt_lower
        if is_realistic:
            realistic_blocks = {"realistic", "photorealistic", "photograph", "real-life"}
            neg_tags = [t.strip() for t in neg.split(",") if t.strip()]
            cleaned_neg = [t for t in neg_tags if t.lower() not in realistic_blocks]
            neg = ", ".join(cleaned_neg)
            
        return pos, neg

    def translate_and_build_prompt(
        self,
        user_description: str,
        style_preset: str = "",
        custom_style: str = "",
        is_vietnamese: bool = True,
    ) -> Tuple[str, str, List[str], str]:
        """
        Translates description and combines with preset.
        Returns: (processed_prompt_en, negative_prompt, matched_char_slugs, primary_char_slug)
        """
        # Match characters
        matched_chars = self.match_characters_in_text(user_description)
        matched_char_slugs = [c.slug for c in matched_chars]
        primary_char_slug = matched_char_slugs[0] if matched_char_slugs else ""
        
        # Load checkpoint to adapt style
        from app.config import load_storytelling_config
        st_config = load_storytelling_config()
        provider = st_config.get("image_gen_provider", "Stable Diffusion (Local GPU)")
        if provider == "Local Gemini (API)":
            checkpoint = ""
        else:
            checkpoint = self.ctx_mgr.load_context().checkpoint if self.ctx_mgr else ""
        
        # Load style
        pos_style = ""
        neg_style = ""
        if style_preset:
            pos_style, neg_style = self.load_style_preset(style_preset)
        elif custom_style:
            pos_style = custom_style
            neg_style = "(worst quality:2), (low quality:2)"
            
        # Adapt style to model
        pos_style, neg_style = self.adapt_style_to_model(pos_style, neg_style, checkpoint)
            
        # Prepare character info for LLM (cleaning portrait tags)
        char_info = []
        for c in matched_chars:
            cleaned_kws = self.clean_portrait_tags(c.keywords_en)
            char_info.append({"name": c.name, "keywords": cleaned_kws})
            
        user_msg = f"Description: {user_description}\n"
        user_msg += f"Style Positive: {pos_style}\n"
        user_msg += f"Style Negative: {neg_style}\n"
        user_msg += f"Matched Characters: {json.dumps(char_info, ensure_ascii=False)}"
        
        messages = [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        result = self._call_llm_json(messages)
        
        final_pos = result.get("positive", "").strip()
        final_neg = result.get("negative", "").strip()
        
        # Fallback if LLM fails
        if not final_pos:
            final_pos = f"{pos_style}, {user_description}".strip(", ")
            final_neg = neg_style

        # SAFETY NET: prompt vượt 77 token đã có compel xử lý (không bị cắt),
        # nhưng vẫn chặn trần 75 từ để tránh LLM viết lan man làm loãng composition.
        final_pos = self._clamp_prompt_words(final_pos, max_words=75)
        final_neg = self._clamp_prompt_words(final_neg, max_words=75)

        return final_pos, final_neg, matched_char_slugs, primary_char_slug

    @staticmethod
    def _clamp_prompt_words(prompt: str, max_words: int = 55) -> str:
        """Cắt prompt theo ranh giới tag sao cho tổng số từ <= max_words (CLIP limit ~77 token)."""
        if not prompt:
            return prompt
        tags = [t.strip() for t in prompt.split(",") if t.strip()]
        kept, count = [], 0
        for t in tags:
            w = len(t.split())
            if count + w > max_words and kept:
                break
            kept.append(t)
            count += w
        clamped = ", ".join(kept)
        if len(kept) < len(tags):
            logger.warning(f"Prompt dài quá giới hạn CLIP, đã cắt {len(tags) - len(kept)} tag cuối: '{prompt[:60]}...'")
        return clamped
