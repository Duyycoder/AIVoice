# -*- coding: utf-8 -*-
"""StudioPipeline — điều phối render theo lớp cho Trạm 2 khi render_mode="studio".

Luồng mỗi cảnh: plan (nền + lớp nhân vật) → nền (cache theo location) →
mỗi nhân vật sinh riêng trên nền phẳng → chroma-key → ghép lớp → lưu frame.
Cảnh khó (đông nhân vật / tương tác vật lý) → fallback render classic.

`render_plan` KHÔNG phụ thuộc ctx/GPU (nhận render_fn + matter qua tham số) → test
được với ảnh giả. `run_batch` mới nạp Stable Diffusion (import lazy).
"""
import os
import re
import unicodedata
from typing import Callable, List, Optional, Tuple

from loguru import logger
from PIL import Image

from app.config import load_storytelling_config
from app.services.storytelling.models import LayerPlan
from app.services.storytelling.studio.layout_planner import (
    build_layer_plan, needs_classic_fallback)
from app.services.storytelling.studio.background_renderer import (
    BackgroundRenderer, safe_location_id)
from app.services.storytelling.studio.character_renderer import CharacterRenderer
from app.services.storytelling.studio.compositor import composite
from app.services.storytelling.studio.matting import (
    ChromaMatter, GrabCutMatter, Matter, alpha_coverage)

# Cổng kiểm định chroma-key: giữ lại quá ít/quá nhiều ⇒ key hỏng ⇒ fallback.
_MIN_COVERAGE = 0.05
_MAX_COVERAGE = 0.95
# Bỏ tag ép chân dung khỏi mô tả nhân vật (ta muốn full body)
_APPEARANCE_BLACKLIST = {
    "upper body", "looking at viewer", "close up", "portrait", "headshot",
    "face focus", "bust shot", "solo focus",
}
# Khung sinh nhân vật (SD1.5 an toàn ≤768) — dọc để full body + mặt to
_CHAR_SIZE = (512, 768)

_PERSON_PROMPT_RE = re.compile(
    r"\b(?:\d+\s*(?:girls?|boys?|women|woman|men|man|persons?|people)|"
    r"(?:a|an|the)?\s*(?:young|old|adult)?\s*"
    r"(?:girls?|boys?|women|woman|men|man|persons?|people|humans?)|"
    r"girls?|boys?|women|woman|men|man|persons?|people|humans?|crowd|solo|"
    r"male focus|female focus|disciples?|students?|soldiers?|villagers?|"
    r"warriors?|monks?|children)\b",
    re.IGNORECASE,
)
_ACTION_CUE_RE = re.compile(
    r"\b(?:hold(?:ing|s)?|walk(?:ing|s)?|run(?:ning|s)?|sit(?:ting|s)?|"
    r"stand(?:ing|s)?|read(?:ing|s)?|writ(?:ing|es?)?|look(?:ing|s)?|"
    r"speak(?:ing|s)?|smil(?:ing|es?)|cry(?:ing|ies)?|point(?:ing|s)?|"
    r"kneel(?:ing|s)?|cast(?:ing|s)?|wield(?:ing|s)?|fly(?:ing|ies)?|"
    r"reach(?:ing|es)?|turn(?:ing|s)?|gestur(?:ing|es?))\b",
    re.IGNORECASE,
)
_CHARACTER_STYLE_BLOCKLIST = (
    "background", "environment", "scenery", "landscape", "setting",
)


class MatteQualityError(RuntimeError):
    """Chroma-key không đạt cổng chất lượng; caller phải render classic."""

    def __init__(self, slug: str, coverage: float):
        self.slug = slug
        self.coverage = coverage
        super().__init__(f"matte_quality:{slug}:coverage={coverage:.3f}")


class StudioPipeline:
    def __init__(self, ctx_mgr=None, context=None):
        self.ctx_mgr = ctx_mgr
        self.context = context

    # ------------------------------------------------------------------
    # Helpers phân giải nhân vật (thuần)
    # ------------------------------------------------------------------
    @staticmethod
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", s or "").encode(
            "ASCII", "ignore").decode("utf-8").lower().replace(" ", "").replace("_", "")

    @staticmethod
    def _norm_words(s: str) -> str:
        text = unicodedata.normalize("NFKD", s or "").encode(
            "ASCII", "ignore").decode("utf-8").lower()
        return " ".join(re.findall(r"[a-z0-9]+", text))

    def _resolve_slugs(self, scene) -> List[str]:
        """Nhân vật xuất hiện trong cảnh (khớp tên/slug), giữ thứ tự xuất hiện."""
        slugs: List[str] = []
        if not self.context:
            return slugs
        names = list(getattr(scene, "characters_in_scene", []) or [])

        by_key = {}
        for ch in self.context.characters:
            by_key[self._norm(ch.name)] = ch
            by_key[self._norm(ch.slug)] = ch

        # Ưu tiên danh sách LLM và giữ đúng thứ tự xuất hiện của nó.
        for name in names:
            ch = by_key.get(self._norm(name))
            if ch and ch.slug not in slugs:
                slugs.append(ch.slug)

        # Tương thích state/prompt cũ có ghi tên: match theo ranh giới từ, không
        # dùng substring kiểu "Lan" trong "landscape".
        prompt_words = f" {self._norm_words(getattr(scene, 'image_prompt', ''))} "
        prompt_hits = []
        for order, ch in enumerate(self.context.characters):
            positions = []
            for raw in (ch.name, ch.slug):
                phrase = self._norm_words(raw)
                if phrase:
                    pos = prompt_words.find(f" {phrase} ")
                    if pos >= 0:
                        positions.append(pos)
            if positions:
                prompt_hits.append((min(positions), order, ch))
        for _, _, ch in sorted(prompt_hits):
            if ch.slug not in slugs:
                slugs.append(ch.slug)
        primary = getattr(scene, "primary_character", "")
        if primary:
            ch = by_key.get(self._norm(primary))
            if ch is None and self.ctx_mgr:
                ch = self.ctx_mgr.get_character(primary)
            if ch and ch.slug not in slugs:
                slugs.append(ch.slug)
        return slugs

    def _get_character(self, slug: str):
        if not self.context:
            return None
        for ch in self.context.characters:
            if ch.slug == slug:
                return ch
        return None

    def _appearance_for(self, slug: str) -> str:
        ch = self._get_character(slug)
        if not ch:
            return ""
        tags = [t.strip() for t in (ch.keywords_en or "").split(",") if t.strip()]
        kept = [t for t in tags if t.lower() not in _APPEARANCE_BLACKLIST]
        return ", ".join(kept)

    def _unresolved_character_reason(self, scene, slugs: List[str]) -> Optional[str]:
        """Ngăn Studio biến cảnh có người thành nền trống khi mapping tên lỗi."""
        known = set()
        if self.context:
            for ch in self.context.characters:
                known.update((self._norm(ch.name), self._norm(ch.slug)))

        declared = list(getattr(scene, "characters_in_scene", []) or [])
        primary = getattr(scene, "primary_character", "")
        if primary:
            declared.append(primary)
        unresolved = [name for name in declared if self._norm(name) not in known]
        if unresolved:
            return "unresolved_characters:" + ",".join(str(x) for x in unresolved[:3])

        if not slugs:
            prompt = (getattr(scene, "image_prompt", "") or "").lower()
            if _PERSON_PROMPT_RE.search(prompt):
                return "person_prompt_without_resolved_character"
        return None

    def _face_image(self, slug: str) -> Optional[Image.Image]:
        if not self.ctx_mgr:
            return None
        try:
            ref = self.ctx_mgr.get_ref_image_path(slug)
            if ref:
                return Image.open(ref).convert("RGB")
        except Exception as e:
            logger.warning(f"[Studio] Không đọc được ảnh ref '{slug}': {e}")
        return None

    def _face_embedding(self, slug: str):
        if not self.ctx_mgr:
            return None
        try:
            import numpy as np
            p = self.ctx_mgr.get_face_embedding_path(slug)
            if p and os.path.exists(p):
                return np.load(p)
        except Exception as e:
            logger.warning(f"[Studio] Không đọc được embedding '{slug}': {e}")
        return None

    def _try_bootstrap_ref(self, scene_idx: int, slug: str, image: Image.Image) -> bool:
        """Lấy mặt lớn từ lớp nhân vật đầu tiên làm ref cho các cảnh sau."""
        if not self.ctx_mgr or self.ctx_mgr.get_ref_image_path(slug):
            return False
        try:
            from app.services.storytelling.face_detailer import detect_faces, _expand_box
            faces = detect_faces(image)
            if not faces:
                return False
            fx, fy, fw, fh = faces[0]
            if min(fw, fh) < 110:
                return False
            img_w, img_h = image.size
            box = _expand_box(fx, fy, fw, fh, img_w, img_h, pad_ratio=0.7)
            face_crop = image.crop(box)
            if face_crop.size[0] < 512 or face_crop.size[1] < 512:
                face_crop = face_crop.resize((512, 512))
            ref_path = self.ctx_mgr.set_ref_from_image(
                slug, face_crop, source="auto_bootstrap")
            if ref_path:
                logger.info(f"[Studio][Bootstrap] Cảnh {scene_idx}: '{slug}' → {ref_path}")
                return True
        except Exception as e:
            logger.warning(f"[Studio][Bootstrap] Bỏ qua '{slug}': {e}")
        return False

    def _scene_location(self, scene) -> str:
        meta = getattr(scene, "_semantic_meta", None)
        if isinstance(meta, dict) and meta.get("location"):
            # Cùng nơi nhưng ngày/đêm khác nhau không được dùng nhầm một nền.
            key = "_".join(str(x) for x in (
                meta["location"], meta.get("time_of_day", "")) if x)
            return safe_location_id(key)
        # Không có location semantic → không tái dùng (mỗi cảnh 1 nền)
        return f"scene_{getattr(scene, 'scene_id', 0):03d}"

    def _background_prompt(self, scene) -> str:
        llm_bg = getattr(scene, "_llm_background_prompt", None)
        if llm_bg and str(llm_bg).strip():
            base = str(llm_bg).strip()
        else:
            meta = getattr(scene, "_semantic_meta", None)
            if isinstance(meta, dict) and meta.get("location"):
                base = ", ".join(str(value).strip() for value in (
                    meta.get("location"), meta.get("time_of_day")) if value)
            else:
                # State cũ có thể chưa có background_prompt. Lọc tag nhân vật
                # khỏi image_prompt để tránh nền tự vẽ người rồi Studio ghép đè.
                appearance_terms = set()
                name_terms = set()
                if self.context:
                    for character in self.context.characters:
                        name_terms.update(self._norm_words(value) for value in (
                            character.name, character.slug) if value)
                        appearance_terms.update(
                            self._norm_words(tag) for tag in
                            (character.keywords_en or "").split(",") if tag.strip())
                kept = []
                for tag in (getattr(scene, "image_prompt", "") or "").split(","):
                    tag = tag.strip()
                    words = self._norm_words(tag)
                    padded = f" {words} "
                    has_name = any(f" {name} " in padded for name in name_terms if name)
                    if (tag and not _PERSON_PROMPT_RE.search(tag) and not has_name
                            and words not in appearance_terms):
                        kept.append(tag)
                base = ", ".join(kept) or "environment background"

        try:
            style = (self.context.get_positive_prompt() if self.context else "").strip().strip(",")
            if style and style.lower() not in base.lower():
                base = f"{style}, {base}" if base else style
        except Exception:
            pass
            
        tags = [t.strip() for t in base.split(",")]
        for t in ["no humans", "no people", "scenery", "empty background"]:
            if t not in tags:
                tags.append(t)
        return ", ".join(tags)

    def _character_style_prompt(self) -> str:
        """Giữ art direction, bỏ tag phong cảnh mâu thuẫn nền chroma phẳng."""
        try:
            raw = self.context.get_positive_prompt() if self.context else ""
        except Exception:
            raw = ""
        return ", ".join(
            tag.strip() for tag in (raw or "").split(",")
            if tag.strip() and not any(
                blocked in tag.lower() for blocked in _CHARACTER_STYLE_BLOCKLIST))

    @staticmethod
    def _image_action_cues(scene) -> str:
        """Lấy action tiếng Anh từ image_prompt cũ khi LLM layout chưa có pose."""
        cues = [tag.strip() for tag in
                (getattr(scene, "image_prompt", "") or "").split(",")
                if _ACTION_CUE_RE.search(tag)]
        return ", ".join(cues[:2])

    def plan_scene(self, scene, slugs: List[str]) -> LayerPlan:
        llm_layout = getattr(scene, "_llm_layout", None)
        fallback_action = self._image_action_cues(scene) if len(slugs) == 1 else ""
        style = self._character_style_prompt()
        chars = []
        for slug in slugs:
            ch = self._get_character(slug)
            display_name = getattr(ch, "name", "") if ch else ""
            has_specific_pose = False
            if isinstance(llm_layout, list):
                aliases = {self._norm(slug), self._norm(display_name)}
                has_specific_pose = any(
                    isinstance(item, dict)
                    and self._norm(item.get("name", "")) in aliases
                    and str(item.get("prompt") or item.get("pose") or "").strip()
                    for item in llm_layout)
            chars.append({
                "slug": slug,
                "name": display_name,
                "prompt": ", ".join(value for value in (
                    "" if has_specific_pose else fallback_action,
                    self._appearance_for(slug), style) if value),
            })
        bg_prompt = self._background_prompt(scene)
        loc_id = self._scene_location(scene)
        
        cfg = load_storytelling_config()
        source = cfg.get("studio_layout_source", "heuristic")
        
        if source == "llm":
            if llm_layout:
                from app.services.storytelling.studio.layout_planner import build_layer_plan_from_llm
                plan = build_layer_plan_from_llm(
                    llm_layout, chars, bg_prompt, loc_id,
                    shot_type=getattr(scene, "shot_type", "wide"))
                if plan:
                    return plan
                else:
                    from loguru import logger
                    logger.warning("[Studio] Parsing LLM layout hỏng → fallback heuristic.")
                    
        # Dù vị trí dùng heuristic, vẫn tận dụng pose/action riêng mà LLM đã trả.
        if isinstance(llm_layout, list):
            poses = {}
            for item in llm_layout:
                if isinstance(item, dict) and item.get("name"):
                    pose = str(item.get("prompt") or item.get("pose") or "").strip().strip(",")
                    if pose:
                        poses[self._norm(item["name"])] = pose
            for char in chars:
                pose = poses.get(self._norm(char.get("name") or char["slug"]))
                if pose:
                    char["prompt"] = ", ".join(x for x in (pose, char["prompt"]) if x)

        return build_layer_plan(
            shot_type=getattr(scene, "shot_type", "wide"),
            chars=chars,
            background_prompt=bg_prompt,
            location_id=loc_id,
        )

    # ------------------------------------------------------------------
    # Render 1 cảnh từ LayerPlan (thuần — test được với render_fn giả)
    # ------------------------------------------------------------------
    def render_plan(self,
                    plan: LayerPlan,
                    size: Tuple[int, int],
                    out_path: str,
                    *,
                    bg_render_fn: Callable[[str, Tuple[int, int]], Image.Image],
                    char_render_fn: Callable[[object], Image.Image],
                    matter: Matter,
                    bg_renderer: BackgroundRenderer,
                    fallback_matter: Optional[Matter] = None,
                    harmonize: bool = True,
                    shadow_opacity: float = 0.0) -> None:
        bg = bg_renderer.get_or_render(plan.location_id, plan.background_prompt, size, bg_render_fn)
        if bg.size != size:
            bg = bg.resize(size)

        layers = []
        for layer in plan.characters:
            raw = char_render_fn(layer)          # RGB trên nền phẳng
            rgba = matter.cutout(raw)            # tách nền → RGBA
            cov = alpha_coverage(rgba)
            if ((cov < _MIN_COVERAGE or cov > _MAX_COVERAGE)
                    and fallback_matter is not None):
                logger.info(f"[Studio] Chroma không đạt cho '{layer.slug}' "
                            f"(coverage={cov:.3f}) — thử matte thích nghi.")
                rgba = fallback_matter.cutout(raw)
                cov = alpha_coverage(rgba)
            if cov < _MIN_COVERAGE or cov > _MAX_COVERAGE:
                raise MatteQualityError(layer.slug, cov)
            layers.append((rgba, layer))

        frame = composite(bg, layers, harmonize=harmonize, shadow_opacity=shadow_opacity)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        frame.save(out_path)

    # ------------------------------------------------------------------
    # Fallback classic cho 1 cảnh (dùng lại engine sinh ảnh gốc)
    # ------------------------------------------------------------------
    def _render_classic_single(self, pipe, scene, out_path: str,
                               size: Tuple[int, int], cfg: dict) -> None:
        slugs = self._resolve_slugs(scene)
        primary = next((s for s in slugs if self.ctx_mgr and self.ctx_mgr.has_identity(s)), None)
        face_img = self._face_image(primary) if primary else None
        face_emb = self._face_embedding(primary) if primary else None
        if hasattr(pipe, "set_character_lora"):
            pipe.set_character_lora(primary)

        shot_map = {"close": (576, 704), "medium": (704, 528), "wide": size}
        w, h = shot_map.get(getattr(scene, "shot_type", "wide"), size)

        img, seed = pipe.generate_draft(
            prompt=scene.image_prompt,
            negative_prompt=self.context.get_negative_prompt(),
            face_embedding=face_emb, face_image=face_img,
            seed=-1, width=w, height=h,
        )
        if cfg.get("enable_face_detailer", True):
            from app.services.storytelling.face_detailer import detail_faces
            img = detail_faces(
                pipe, img, prompt=scene.image_prompt,
                negative_prompt=self.context.get_negative_prompt(),
                face_image=face_img, face_embedding=face_emb,
                strength=cfg.get("face_detailer_strength", 0.45),
            )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
        scene.accepted_seed = seed

    # ------------------------------------------------------------------
    # Chạy cả batch (nạp Stable Diffusion)
    # ------------------------------------------------------------------
    def _detect_lead_slugs(self, scenes: List, max_leads: int) -> List[str]:
        """Nhân vật chính = xuất hiện ở NHIỀU cảnh nhất (nam/nữ chính)."""
        from collections import Counter
        counts: Counter = Counter()
        for scene in scenes:
            for slug in self._resolve_slugs(scene):
                counts[slug] += 1
        return [slug for slug, _ in counts.most_common(max_leads)]

    def _auto_train_leads(self, scenes: List, cfg: dict,
                          progress_cb: Optional[Callable[[str, int], None]]) -> None:
        """Bootstrap + train LoRA cho nhân vật chính 1 lần/truyện (idempotent)."""
        from app.services.storytelling.character_bootstrap import (
            auto_train_leads, has_trained_lora)
        max_leads = int(cfg.get("studio_auto_train_max_leads", 2))
        leads = self._detect_lead_slugs(scenes, max_leads)
        checkpoint = getattr(self.context, "checkpoint", "") or ""
        pending = [s for s in leads if not has_trained_lora(s, checkpoint)]
        if not pending:
            if leads:
                logger.info(f"[Studio] Nhân vật chính {leads} đã có LoRA — bỏ qua train.")
            return
        logger.info(f"[Studio] Auto-train LoRA nhân vật chính (1 lần/truyện): {pending}")
        auto_train_leads(
            self.ctx_mgr, lead_slugs=pending, max_leads=max_leads,
            steps=int(cfg.get("studio_auto_train_steps", 700)),
            progress_cb=(lambda m: progress_cb(m, 10)) if progress_cb else None)

    def run_batch(self, scenes: List, out_dir: str,
                  progress_cb: Optional[Callable[[str, int], None]] = None) -> List:
        cfg = load_storytelling_config()
        os.makedirs(out_dir, exist_ok=True)

        # Giải phóng LLM local trước khi nạp SD (giống classic)
        try:
            from app.services.llm import unload_local_llm
            unload_local_llm()
        except Exception:
            pass

        # Train LoRA nhân vật chính 1 LẦN cho truyện (idempotent). Làm TRƯỚC khi nạp
        # SD để render vì train_character sẽ release pipeline; sau đó warmup lại sạch.
        if bool(cfg.get("studio_auto_train_leads", True)) and self.ctx_mgr and self.context:
            try:
                self._auto_train_leads(scenes, cfg, progress_cb)
            except Exception as e:
                logger.warning(f"[Studio] auto_train_leads bỏ qua ({e}) — render không LoRA.")

        from app.services.storytelling.image_generator import StorytellingPipeline
        pipe = StorytellingPipeline(self.context)
        pipe.warmup()
        if hasattr(pipe, "update_ip_adapter_scale"):
            pipe.update_ip_adapter_scale(float(cfg.get("ip_adapter_scale", 0.6)))

        size = (cfg.get("image_width", 768), cfg.get("image_height", 432))
        bg_hex = cfg.get("studio_matte_bg_color", "#00B140")
        # Engine matte chính: "rembg" (isnet-anime — segment nhân vật, bền nhất) hoặc
        # "chroma" (chroma-key cũ — chỉ tốt khi SD vẽ được nền phẳng đúng màu).
        engine = str(cfg.get("studio_matte_engine", "rembg")).lower()
        chroma = ChromaMatter(
            bg_color=bg_hex,
            threshold=float(cfg.get("studio_matte_threshold", 0.18)),
            feather_px=int(cfg.get("studio_matte_feather_px", 3)),
            despill=bool(cfg.get("studio_matte_despill", True)),
        )
        grabcut = None
        if bool(cfg.get("studio_matte_adaptive_fallback", True)):
            grabcut = GrabCutMatter(
                feather_px=int(cfg.get("studio_matte_feather_px", 3)))
        if engine == "rembg":
            try:
                from app.services.storytelling.studio.matting import RembgMatter
                matter = RembgMatter(
                    model_name=str(cfg.get("studio_matte_model", "isnet-anime")),
                    feather_px=int(cfg.get("studio_matte_feather_px", 2)))
                # Rembg bền hơn chroma nên xếp chroma→grabcut làm lưới an toàn.
                fallback_matter = grabcut or chroma
                logger.info("[Studio] Matte engine: rembg (isnet-anime).")
            except Exception as e:
                logger.warning(f"[Studio] Rembg không sẵn sàng ({e}) — dùng chroma-key.")
                matter, fallback_matter = chroma, grabcut
        else:
            matter, fallback_matter = chroma, grabcut
        bg_cache_dir = os.path.join(out_dir, "..", "bg_cache")
        if self.ctx_mgr and getattr(self.ctx_mgr, "context_dir", ""):
            bg_cache_dir = os.path.join(self.ctx_mgr.context_dir, "bg_cache")
        bg_renderer = BackgroundRenderer(
            os.path.abspath(bg_cache_dir), enabled=bool(cfg.get("studio_bg_cache", True)))
        char_renderer = CharacterRenderer()
        use_ip = bool(cfg.get("studio_char_use_ip_adapter", True))
        use_detailer = bool(cfg.get("studio_char_use_detailer", True))
        # Nhân vật đã có LoRA giữ nhận dạng → bỏ detailer cho nhanh (config bật/tắt).
        lora_skip_detailer = bool(cfg.get("studio_lora_skip_detailer", True))
        _checkpoint = getattr(self.context, "checkpoint", "") or ""
        from app.services.storytelling.character_bootstrap import has_trained_lora

        def bg_render_fn(prompt, sz):
            # Character LoRA là stateful; phải tắt trước khi sinh nền location mới.
            if hasattr(pipe, "set_character_lora"):
                pipe.set_character_lora(None)
            img, _ = pipe.generate_draft(
                prompt=prompt, negative_prompt=self.context.get_negative_prompt(),
                face_embedding=None, face_image=None, seed=-1, width=sz[0], height=sz[1])
            return img

        total = max(len(scenes), 1)
        for i, scene in enumerate(scenes):
            out_path = os.path.join(out_dir, f"scene_{i:03d}.png")
            if progress_cb:
                progress_cb(f"Studio: cảnh {i + 1}/{len(scenes)}",
                            20 + int(70 * i / total))
            try:
                slugs = self._resolve_slugs(scene)
                reason = self._unresolved_character_reason(scene, slugs)
                if not reason:
                    reason = needs_classic_fallback(
                        len(slugs), getattr(scene, "image_prompt", ""),
                        int(cfg.get("studio_fallback_max_chars", 3)),
                        cfg.get("studio_fallback_interaction_tags", []))
                if reason:
                    logger.info(f"[Studio] Cảnh {i}: fallback classic ({reason}).")
                    self._render_classic_single(pipe, scene, out_path, size, cfg)
                else:
                    plan = self.plan_scene(scene, slugs)

                    def char_render_fn(layer):
                        face_img = self._face_image(layer.slug) if use_ip else None
                        face_emb = self._face_embedding(layer.slug) if use_ip else None
                        if hasattr(pipe, "set_character_lora"):
                            pipe.set_character_lora(layer.slug)
                        # LoRA đã giữ nhận dạng → bỏ detailer cho nhân vật này.
                        layer_detailer = use_detailer
                        if lora_skip_detailer and has_trained_lora(layer.slug, _checkpoint):
                            layer_detailer = False
                        img, _ = char_renderer.render(
                            pipe, layer.prompt, _CHAR_SIZE, bg_hex,
                            face_image=face_img, face_embedding=face_emb,
                            negative_prompt=self.context.get_negative_prompt(),
                            use_detailer=layer_detailer,
                            framing=getattr(layer, "framing", "full"))
                        if use_ip and face_img is None and face_emb is None:
                            self._try_bootstrap_ref(i, layer.slug, img)
                        return img

                    self.render_plan(plan, size, out_path,
                                     bg_render_fn=bg_render_fn,
                                     char_render_fn=char_render_fn,
                                     matter=matter, fallback_matter=fallback_matter,
                                     bg_renderer=bg_renderer,
                                     shadow_opacity=float(cfg.get("studio_shadow_opacity", 0.0)))
            except Exception as e:
                logger.error(f"[Studio] Cảnh {i} lỗi ({e}) — thử fallback classic.")
                try:
                    self._render_classic_single(pipe, scene, out_path, size, cfg)
                except Exception as e2:
                    logger.exception(f"[Studio] Fallback classic cũng lỗi: {e2}")
                    raise RuntimeError(
                        f"Studio và fallback classic đều lỗi ở cảnh {i}: {e2}") from e2
            scene.frame_path = out_path

        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return scenes
