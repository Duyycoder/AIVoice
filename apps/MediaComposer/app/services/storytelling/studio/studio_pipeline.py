# -*- coding: utf-8 -*-
"""StudioPipeline — điều phối render theo lớp cho Trạm 2 khi render_mode="studio".

Luồng mỗi cảnh: plan (nền + lớp nhân vật) → nền (cache theo location) →
mỗi nhân vật sinh riêng trên nền phẳng → chroma-key → ghép lớp → lưu frame.
Cảnh khó (đông nhân vật / tương tác vật lý) → fallback render classic.

`render_plan` KHÔNG phụ thuộc ctx/GPU (nhận render_fn + matter qua tham số) → test
được với ảnh giả. `run_batch` mới nạp Stable Diffusion (import lazy).
"""
import os
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
from app.services.storytelling.studio.character_renderer import (
    CharacterRenderer, bg_color_name)
from app.services.storytelling.studio.compositor import composite
from app.services.storytelling.studio.matting import ChromaMatter, alpha_coverage

# Cổng kiểm định chroma-key: giữ lại quá ít/quá nhiều ⇒ key hỏng ⇒ bỏ lớp.
_MIN_COVERAGE = 0.02
_MAX_COVERAGE = 0.98
# Bỏ tag ép chân dung khỏi mô tả nhân vật (ta muốn full body)
_APPEARANCE_BLACKLIST = {
    "upper body", "looking at viewer", "close up", "portrait", "headshot",
    "face focus", "bust shot", "solo focus",
}
# Khung sinh nhân vật (SD1.5 an toàn ≤768) — dọc để full body + mặt to
_CHAR_SIZE = (512, 768)


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

    def _resolve_slugs(self, scene) -> List[str]:
        """Nhân vật xuất hiện trong cảnh (khớp tên/slug), giữ thứ tự xuất hiện."""
        slugs: List[str] = []
        if not self.context:
            return slugs
        pn = self._norm(getattr(scene, "image_prompt", ""))
        names = list(getattr(scene, "characters_in_scene", []) or [])
        for ch in self.context.characters:
            matched = (any(self._norm(n) == self._norm(ch.name) for n in names)
                       or self._norm(ch.name) in pn or self._norm(ch.slug) in pn)
            if matched and ch.slug not in slugs:
                slugs.append(ch.slug)
        primary = getattr(scene, "primary_character", "")
        if primary and self.ctx_mgr:
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

    def _scene_location(self, scene) -> str:
        meta = getattr(scene, "_semantic_meta", None)
        if isinstance(meta, dict) and meta.get("location"):
            return safe_location_id(meta["location"])
        # Không có location semantic → không tái dùng (mỗi cảnh 1 nền)
        return f"scene_{getattr(scene, 'scene_id', 0):03d}"

    def _background_prompt(self, scene) -> str:
        base = (getattr(scene, "image_prompt", "") or "").strip().strip(",")
        return f"{base}, no humans, no people, scenery, empty background".strip(", ")

    def plan_scene(self, scene, slugs: List[str]) -> LayerPlan:
        chars = [{"slug": s, "prompt": self._appearance_for(s)} for s in slugs]
        return build_layer_plan(
            shot_type=getattr(scene, "shot_type", "wide"),
            chars=chars,
            background_prompt=self._background_prompt(scene),
            location_id=self._scene_location(scene),
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
                    matter: ChromaMatter,
                    bg_renderer: BackgroundRenderer,
                    harmonize: bool = True) -> None:
        bg = bg_renderer.get_or_render(plan.location_id, plan.background_prompt, size, bg_render_fn)
        if bg.size != size:
            bg = bg.resize(size)

        layers = []
        for layer in plan.characters:
            raw = char_render_fn(layer)          # RGB trên nền phẳng
            rgba = matter.cutout(raw)            # tách nền → RGBA
            cov = alpha_coverage(rgba)
            if cov < _MIN_COVERAGE or cov > _MAX_COVERAGE:
                logger.warning(f"[Studio] Bỏ lớp '{layer.slug}': chroma-key hỏng "
                               f"(coverage={cov:.3f}).")
                continue
            layers.append((rgba, layer))

        frame = composite(bg, layers, harmonize=harmonize)
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

        from app.services.storytelling.image_generator import StorytellingPipeline
        pipe = StorytellingPipeline(self.context)
        pipe.warmup()
        if hasattr(pipe, "update_ip_adapter_scale"):
            pipe.update_ip_adapter_scale(float(cfg.get("ip_adapter_scale", 0.6)))

        size = (cfg.get("image_width", 768), cfg.get("image_height", 432))
        bg_hex = cfg.get("studio_matte_bg_color", "#00B140")
        matter = ChromaMatter(
            bg_color=bg_hex,
            threshold=float(cfg.get("studio_matte_threshold", 0.18)),
            feather_px=int(cfg.get("studio_matte_feather_px", 3)),
            despill=bool(cfg.get("studio_matte_despill", True)),
        )
        bg_cache_dir = os.path.join(out_dir, "..", "bg_cache")
        bg_renderer = BackgroundRenderer(
            os.path.abspath(bg_cache_dir), enabled=bool(cfg.get("studio_bg_cache", True)))
        char_renderer = CharacterRenderer()
        use_ip = bool(cfg.get("studio_char_use_ip_adapter", True))
        use_detailer = bool(cfg.get("studio_char_use_detailer", True))

        def bg_render_fn(prompt, sz):
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
                reason = needs_classic_fallback(
                    len(slugs), getattr(scene, "image_prompt", ""),
                    int(cfg.get("studio_fallback_max_chars", 3)),
                    cfg.get("studio_fallback_interaction_tags", []))
                if reason:
                    logger.info(f"[Studio] Cảnh {i}: fallback classic ({reason}).")
                    self._render_classic_single(pipe, scene, out_path, size, cfg)
                else:
                    plan = self.plan_scene(scene, slugs)

                    def char_render_fn(layer, _slugs=slugs):
                        face_img = self._face_image(layer.slug) if use_ip else None
                        face_emb = self._face_embedding(layer.slug) if use_ip else None
                        if hasattr(pipe, "set_character_lora"):
                            pipe.set_character_lora(layer.slug)
                        img, _ = char_renderer.render(
                            pipe, layer.prompt, _CHAR_SIZE, bg_hex,
                            face_image=face_img, face_embedding=face_emb,
                            negative_prompt=self.context.get_negative_prompt(),
                            use_detailer=use_detailer)
                        return img

                    self.render_plan(plan, size, out_path,
                                     bg_render_fn=bg_render_fn,
                                     char_render_fn=char_render_fn,
                                     matter=matter, bg_renderer=bg_renderer)
            except Exception as e:
                logger.error(f"[Studio] Cảnh {i} lỗi ({e}) — thử fallback classic.")
                try:
                    self._render_classic_single(pipe, scene, out_path, size, cfg)
                except Exception as e2:
                    logger.error(f"[Studio] Fallback classic cũng lỗi: {e2} — lưu ảnh xám.")
                    Image.new("RGB", size, (40, 40, 40)).save(out_path)
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
