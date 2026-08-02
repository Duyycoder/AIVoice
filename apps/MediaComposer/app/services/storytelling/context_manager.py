import os
import json
import datetime
import dataclasses
from typing import List, Tuple
from dataclasses import asdict
from loguru import logger

from app.services.storytelling.models import StoryContext, Character, LearnedCorrections

# Đường dẫn tuyệt đối theo MediaComposer root — không phụ thuộc CWD
_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_STORAGE_ENV = os.getenv("MC_STORAGE_TASKS")
if _STORAGE_ENV:
    CONTEXTS_ROOT = os.path.join(os.path.abspath(_STORAGE_ENV), "contexts")
else:
    CONTEXTS_ROOT = os.path.join(_MC_ROOT, "storage", "contexts")

class ContextManager:
    def __init__(self, story_slug: str):
        self.story_slug = story_slug
        self.context_dir = os.path.join(CONTEXTS_ROOT, story_slug)
        self.context_file = os.path.join(self.context_dir, "context.json")
        self.style_file = os.path.join(self.context_dir, "style_prompt.txt")
        self.learned_file = os.path.join(self.context_dir, "learned_corrections.json")
        self.chars_dir = os.path.join(self.context_dir, "characters")
        
        self._context: StoryContext = None

    # Preset dùng khi tạo truyện mới và khi UI không chỉ định gì khác.
    DEFAULT_STYLE_PRESET = "thuy_mac"

    _FALLBACK_STYLE = (
        "(masterpiece, best quality:1.2), ink wash painting, monochrome grayscale, "
        "heavy black ink, bold silhouette, high contrast, misty atmosphere\n"
        "---\n"
        "(worst quality:2), (low quality:2), lowres, (bad anatomy:1.4), "
        "(bad hands:1.5), (mutated hands:1.4), text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, "
        "username, (extra limbs:1.4), (deformed:1.3), blurry, bad face, "
        "realistic, 3D render, photograph, photorealistic, nsfw, "
        "western cartoon, out of frame"
    )

    def _read_default_preset(self) -> str:
        """Nội dung preset mặc định; thiếu file thì dùng chuỗi dự phòng cùng tinh thần."""
        preset_path = os.path.join(
            _MC_ROOT, "resource", "image_presets",
            f"{self.DEFAULT_STYLE_PRESET}.txt")
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return self._FALLBACK_STYLE

    def apply_style_preset(self, preset_name: str) -> bool:
        """Chép preset trong resource/image_presets vào style file của truyện.

        Trước đây `adapter_video_cli` chỉ gán `context.art_style = <tên>` — một
        chuỗi thuần, không ai đọc để đổi prompt. Style THẬT được ghi đúng một lần
        lúc `create_context()` rồi giữ nguyên vĩnh viễn, nên chọn phong cách trên
        UI hoàn toàn không có tác dụng lên ảnh. Hàm này nối lại mắt xích đó.

        Trả False nếu không có preset tương ứng (giữ nguyên style hiện tại).
        """
        if not preset_name:
            return False
        preset_path = os.path.join(
            _MC_ROOT, "resource", "image_presets", f"{preset_name}.txt")
        if not os.path.exists(preset_path):
            logger.warning(
                f"[Context] Không có preset '{preset_name}' — giữ style hiện tại.")
            return False
        try:
            with open(preset_path, "r", encoding="utf-8") as src:
                content = src.read()
            os.makedirs(self.context_dir, exist_ok=True)
            with open(self.style_file, "w", encoding="utf-8") as dst:
                dst.write(content)
            logger.info(f"[Context] Đã áp phong cách '{preset_name}' cho truyện "
                        f"'{self.story_slug}'.")
            return True
        except OSError as e:
            logger.warning(f"[Context] Không áp được preset '{preset_name}': {e}")
            return False

    def create_context(self, story_name: str, genre: str) -> StoryContext:
        os.makedirs(self.chars_dir, exist_ok=True)

        default_style = self._read_default_preset()
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
        valid_fields = {f.name for f in dataclasses.fields(Character)}
        for c in data.get("characters", []):
            filtered = {k: v for k, v in c.items() if k in valid_fields}
            chars.append(Character(**filtered))
            
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
        
        if ref_image_path and os.path.exists(ref_image_path):
            import shutil
            ext = os.path.splitext(ref_image_path)[1]
            shutil.copy(ref_image_path, os.path.join(char_dir, f"ref{ext}"))

        # FIX: không reset has_embedding=False khi cập nhật nhân vật đã có embedding.
        # Nguồn sự thật là file face.ipadpt.npy trên đĩa.
        has_embedding = os.path.exists(self.get_face_embedding_path(slug))

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

    def get_face_embedding_path(self, slug: str) -> str:
        """Đường dẫn chuẩn của face embedding cho một nhân vật (nguồn sự thật duy nhất)."""
        return os.path.join(self.chars_dir, slug, "face.ipadpt.npy")

    def has_face_embedding(self, slug: str) -> bool:
        """Check thống nhất cho mọi luồng (Studio ảnh & Workflow video): dựa trên file thực tế."""
        return bool(slug) and os.path.exists(self.get_face_embedding_path(slug))

    def get_ref_image_path(self, slug: str) -> str:
        """Đường dẫn ảnh tham chiếu gốc (ref.*) của nhân vật — dùng cho IP-Adapter CLIP."""
        if not slug:
            return ""
        char_dir = os.path.join(self.chars_dir, slug)
        if os.path.isdir(char_dir):
            for f in sorted(os.listdir(char_dir)):
                name, ext = os.path.splitext(f)
                if name == "ref" and ext.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    return os.path.join(char_dir, f)
        return ""

    def set_ref_from_image(self, slug: str, image, source: str = "auto_bootstrap") -> str:
        """T7: đặt ảnh ref cho nhân vật từ PIL Image (bootstrap từ batch).

        - KHÔNG ghi đè ref do user upload tay (chỉ set khi chưa có ref,
          hoặc ref hiện tại cũng là auto_bootstrap).
        - Xoá cache .ref_emb.npy để similarity tính lại theo ref mới.
        Trả đường dẫn ref đã lưu, hoặc chuỗi rỗng nếu bị từ chối.
        """
        char = self.get_character(slug)
        if not char:
            return ""
        existing = self.get_ref_image_path(slug)
        if existing and getattr(char, "ref_source", "manual") == "manual":
            return ""  # tôn trọng ref user đặt tay

        char_dir = os.path.join(self.chars_dir, slug)
        os.makedirs(char_dir, exist_ok=True)
        # Xoá ref cũ khác định dạng để get_ref_image_path không trả file cũ
        for f in os.listdir(char_dir):
            name, ext = os.path.splitext(f)
            if name == "ref" and ext.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                try:
                    os.remove(os.path.join(char_dir, f))
                except Exception:
                    pass
        ref_path = os.path.join(char_dir, "ref.png")
        image.save(ref_path, format="PNG")

        # Invalidate cache embedding của ref
        emb_cache = os.path.join(self.get_dataset_dir(slug), ".ref_emb.npy")
        if os.path.exists(emb_cache):
            try:
                os.remove(emb_cache)
            except Exception:
                pass

        char.ref_source = source
        self.save_context()
        return ref_path

    def has_identity(self, slug: str) -> bool:
        """Nhân vật có dữ liệu nhận dạng (ảnh tham chiếu HOẶC face embedding)."""
        return bool(self.get_ref_image_path(slug)) or self.has_face_embedding(slug)

    def get_character(self, slug: str) -> Character:
        if not self._context:
            self.load_context()
        for c in self._context.characters:
            if c.slug == slug:
                return c
        return None

    def delete_character(self, slug: str) -> bool:
        if not self._context:
            self.load_context()
        initial_len = len(self._context.characters)
        self._context.characters = [c for c in self._context.characters if c.slug != slug]
        if len(self._context.characters) < initial_len:
            self.save_context()
            
            # Optionally delete character directory
            char_dir = os.path.join(self.chars_dir, slug)
            if os.path.exists(char_dir):
                import shutil
                try:
                    shutil.rmtree(char_dir)
                except Exception:
                    pass
            return True
        return False

    def list_characters(self) -> List[Character]:
        if not self._context:
            self.load_context()
        return self._context.characters

    def get_style_prompt(self) -> Tuple[str, str]:
        if not self._context:
            self.load_context()
        return self._context.get_positive_prompt(), self._context.get_negative_prompt()

    def get_dataset_dir(self, slug: str) -> str:
        """Thư mục dataset ảnh chuẩn hoá cho train LoRA."""
        ds_dir = os.path.join(self.chars_dir, slug, "dataset")
        os.makedirs(ds_dir, exist_ok=True)
        return ds_dir

    def count_dataset_images(self, slug: str) -> int:
        """Đếm ảnh approved_* + auto_* trong dataset."""
        ds_dir = self.get_dataset_dir(slug)
        count = 0
        for f in os.listdir(ds_dir):
            if (f.startswith("approved_") or f.startswith("auto_")) and f.endswith(".png"):
                count += 1
        return count

    def add_dataset_image(self, slug: str, image_path_or_pil, source: str) -> str:
        """
        Lưu ảnh vào dataset. source: "approved" | "auto".
        Trả path ảnh đã lưu. Áp trần 40 ảnh — khi vượt, xoá auto_* cũ nhất (FIFO).
        KHÔNG bao giờ tự xoá approved_*.
        """
        import uuid as _uuid
        from PIL import Image as PILImage

        ds_dir = self.get_dataset_dir(slug)
        prefix = "approved" if source == "approved" else "auto"
        filename = f"{prefix}_{_uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(ds_dir, filename)

        # Save image
        if isinstance(image_path_or_pil, PILImage.Image):
            image_path_or_pil.save(out_path, format="PNG")
        else:
            import shutil
            shutil.copy2(str(image_path_or_pil), out_path)

        # Enforce cap: max 40 images, evict oldest auto_* first
        MAX_DATASET_IMAGES = 40
        all_imgs = sorted(
            [f for f in os.listdir(ds_dir)
             if (f.startswith("approved_") or f.startswith("auto_")) and f.endswith(".png")],
            key=lambda f: os.path.getmtime(os.path.join(ds_dir, f))
        )
        while len(all_imgs) > MAX_DATASET_IMAGES:
            auto_imgs = [f for f in all_imgs if f.startswith("auto_")]
            if auto_imgs:
                victim = auto_imgs[0]
                os.remove(os.path.join(ds_dir, victim))
                all_imgs.remove(victim)
            else:
                break  # Only approved_* left — never auto-delete those

        return out_path

    @staticmethod
    def list_all_contexts() -> List[str]:
        if not os.path.exists(CONTEXTS_ROOT):
            return []
        return [d for d in os.listdir(CONTEXTS_ROOT) if os.path.isdir(os.path.join(CONTEXTS_ROOT, d))]
