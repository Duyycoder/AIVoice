from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class Scene:
    scene_id: int           # 0-indexed
    text_vi: str            # Văn bản tiếng Việt gốc
    word_count: int
    start_time: float       # giây, từ SRT
    end_time: float         # giây, từ SRT
    duration_sec: float     # = end_time - start_time
    image_prompt: str       # EN, do LLM sinh
    characters_in_scene: List[str]  # tên nhân vật được detect
    primary_character: str  # nhân vật dùng face embedding
    fallback_level: int     # 0=chưa refine, 1-4=đã thử cấp nào
    accepted_seed: int      # seed của ảnh được chọn
    frame_path: str         # đường dẫn ảnh đã chọn

@dataclass
class Character:
    name: str
    slug: str
    description: str
    keywords_en: str
    has_embedding: bool

@dataclass
class LearnedCorrections:
    version: int = 1
    last_updated: str = ""
    sessions_count: int = 0
    prompt_additions: List[str] = field(default_factory=list)
    prompt_removals: List[str] = field(default_factory=list)
    prompt_substitutions: Dict[str, str] = field(default_factory=dict)
    style_notes: str = ""
    escalation_stats: Dict[str, int] = field(default_factory=lambda: {
        "level_1_attempts": 0, "level_1_accepted": 0,
        "level_2_attempts": 0, "level_2_accepted": 0,
        "level_3_attempts": 0, "level_3_accepted": 0,
        "level_4_attempts": 0, "level_4_accepted": 0
    })
    delta_candidates: List[dict] = field(default_factory=list)

@dataclass
class StoryContext:
    story_name: str
    story_slug: str
    genre: str
    art_style: str = "anime_2d_flat"
    checkpoint: str = "stablediffusionapi/anything-v5"
    created_at: str = ""
    characters: List[Character] = field(default_factory=list)
    learned_corrections: LearnedCorrections = field(default_factory=LearnedCorrections)
    
    _style_prompt_path: str = ""
    
    def get_positive_prompt(self) -> str:
        if not self._style_prompt_path:
            return ""
        try:
            with open(self._style_prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
            base = content.split("---")[0].strip()
            additions = " ".join(self.learned_corrections.prompt_additions)
            return f"{base}, {additions}".strip(", ")
        except Exception:
            return ""

    def get_negative_prompt(self) -> str:
        if not self._style_prompt_path:
            return ""
        try:
            with open(self._style_prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
            parts = content.split("---")
            base = parts[1].strip() if len(parts) > 1 else ""
            removals = " ".join(self.learned_corrections.prompt_removals)
            return f"{base}, {removals}".strip(", ")
        except Exception:
            return ""
