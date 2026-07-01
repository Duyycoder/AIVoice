import os
import json
import datetime
from typing import List, Tuple
from dataclasses import asdict
from loguru import logger

from app.services.storytelling.models import StoryContext, Character, LearnedCorrections

CONTEXTS_ROOT = os.path.join("storage", "contexts")

class ContextManager:
    def __init__(self, story_slug: str):
        self.story_slug = story_slug
        self.context_dir = os.path.join(CONTEXTS_ROOT, story_slug)
        self.context_file = os.path.join(self.context_dir, "context.json")
        self.style_file = os.path.join(self.context_dir, "style_prompt.txt")
        self.learned_file = os.path.join(self.context_dir, "learned_corrections.json")
        self.chars_dir = os.path.join(self.context_dir, "characters")
        
        self._context: StoryContext = None

    def create_context(self, story_name: str, genre: str) -> StoryContext:
        os.makedirs(self.chars_dir, exist_ok=True)
        
        default_style = (
            "anime style, 2D flat illustration, cel shaded, clean lineart, Anything V5, \n"
            "xianxia cultivation setting, traditional Chinese aesthetics, vibrant colors,\n"
            "sharp details, high quality\n"
            "---\n"
            "realistic, 3D render, photograph, photorealistic, nsfw, extra limbs, \n"
            "bad anatomy, low quality, blurry, western cartoon, chibi (trừ khi yêu cầu)"
        )
        with open(self.style_file, "w", encoding="utf-8") as f:
            f.write(default_style)
            
        ctx = StoryContext(
            story_name=story_name,
            story_slug=self.story_slug,
            genre=genre,
            created_at=datetime.datetime.now().isoformat()
        )
        ctx._style_prompt_path = self.style_file
        
        with open(self.learned_file, "w", encoding="utf-8") as f:
            json.dump(asdict(ctx.learned_corrections), f, indent=2, ensure_ascii=False)
            
        self._context = ctx
        self.save_context(ctx)
        return ctx

    def load_context(self) -> StoryContext:
        if not os.path.exists(self.context_file):
            raise FileNotFoundError(f"Context for {self.story_slug} not found.")
            
        with open(self.context_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        chars = []
        for c in data.get("characters", []):
            chars.append(Character(**c))
            
        learned = LearnedCorrections()
        if os.path.exists(self.learned_file):
            with open(self.learned_file, "r", encoding="utf-8") as f:
                l_data = json.load(f)
                for k, v in l_data.items():
                    if hasattr(learned, k):
                        setattr(learned, k, v)
                        
        self._context = StoryContext(
            story_name=data.get("story_name", ""),
            story_slug=data.get("story_slug", ""),
            genre=data.get("genre", ""),
            art_style=data.get("art_style", "anime_2d_flat"),
            checkpoint=data.get("checkpoint", "anything-v5"),
            created_at=data.get("created_at", ""),
            characters=chars,
            learned_corrections=learned
        )
        self._context._style_prompt_path = self.style_file
        return self._context

    def save_context(self, ctx: StoryContext = None) -> None:
        if ctx is None:
            ctx = self._context
        if not ctx:
            return
            
        os.makedirs(self.context_dir, exist_ok=True)
        
        data = {
            "story_name": ctx.story_name,
            "story_slug": ctx.story_slug,
            "genre": ctx.genre,
            "art_style": ctx.art_style,
            "checkpoint": ctx.checkpoint,
            "created_at": ctx.created_at,
            "characters": [asdict(c) for c in ctx.characters]
        }
        
        with open(self.context_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        with open(self.learned_file, "w", encoding="utf-8") as f:
            json.dump(asdict(ctx.learned_corrections), f, indent=2, ensure_ascii=False)

    def add_character(self, name: str, description: str, keywords_en: str, ref_image_path: str = "") -> None:
        if not self._context:
            self.load_context()
            
        import re
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_')
        char_dir = os.path.join(self.chars_dir, slug)
        os.makedirs(char_dir, exist_ok=True)
        
        has_embedding = False
        if ref_image_path and os.path.exists(ref_image_path):
            import shutil
            ext = os.path.splitext(ref_image_path)[1]
            shutil.copy(ref_image_path, os.path.join(char_dir, f"ref{ext}"))
            
        char = Character(
            name=name,
            slug=slug,
            description=description,
            keywords_en=keywords_en,
            has_embedding=has_embedding
        )
        
        existing_idx = next((i for i, c in enumerate(self._context.characters) if c.slug == slug), -1)
        if existing_idx >= 0:
            self._context.characters[existing_idx] = char
        else:
            self._context.characters.append(char)
            
        self.save_context()

    def get_character(self, slug: str) -> Character:
        if not self._context:
            self.load_context()
        for c in self._context.characters:
            if c.slug == slug:
                return c
        return None

    def list_characters(self) -> List[Character]:
        if not self._context:
            self.load_context()
        return self._context.characters

    def get_style_prompt(self) -> Tuple[str, str]:
        if not self._context:
            self.load_context()
        return self._context.get_positive_prompt(), self._context.get_negative_prompt()

    @staticmethod
    def list_all_contexts() -> List[str]:
        if not os.path.exists(CONTEXTS_ROOT):
            return []
        return [d for d in os.listdir(CONTEXTS_ROOT) if os.path.isdir(os.path.join(CONTEXTS_ROOT, d))]
