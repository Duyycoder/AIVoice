import os
import uuid
# Force-load DLLs in exact order: torch -> faster_whisper -> cv2 to prevent Windows C++ CUDA/OpenMP abort
try:
    import torch  # noqa: F401
    import faster_whisper  # noqa: F401
    import cv2  # noqa: F401
except Exception:
    pass
from loguru import logger
from typing import Callable, Optional, Tuple

from app.services.storytelling.context_manager import ContextManager
from app.services.storytelling.md_parser import parse_md_to_scenes
from app.services.storytelling.srt_mapper import map_scenes_to_timeline
from app.services.storytelling.llm_prompter import generate_prompts_batch
from app.services.storytelling.image_generator import StorytellingPipeline
from app.services.storytelling.postprocess import PostProcessor
from app.services.storytelling.video_assembler import assemble_video, FrameInfo

class StorytellingOrchestrator:
    def __init__(self, ctx_mgr: ContextManager):
        self.ctx_mgr = ctx_mgr
        self.context = ctx_mgr.load_context()

    def get_state_path(self) -> str:
        # FIX Gap4: batch isolation — dùng state_path riêng nếu có
        if hasattr(self, '_state_path_override') and self._state_path_override:
            return self._state_path_override
        return os.path.join(self.ctx_mgr.context_dir, "state.json")

    def save_state(self, step: str, scenes: list, task_dir: str, audio_path: str = "", srt_path: str = "", md_path: str = "") -> None:
        import json
        from dataclasses import asdict
        state_data = {
            "step": step,
            "scenes": [asdict(s) for s in scenes],
            "task_dir": task_dir,
            "audio_path": audio_path,
            "srt_path": srt_path,
            "md_path": md_path
        }
        state_path = self.get_state_path()
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)

    def load_state(self) -> Optional[dict]:
        import json
        state_path = self.get_state_path()
        if not os.path.exists(state_path):
            return None
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state from {state_path}: {e}")
            return None

    def clear_state(self) -> None:
        state_path = self.get_state_path()
        if os.path.exists(state_path):
            try:
                os.remove(state_path)
            except Exception:
                pass

    def _resolve_primary_character(self, scene) -> Optional[str]:
        import unicodedata
        def normalize(s):
            return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower().replace(' ', '').replace('_', '')
            
        # FIX: check dữ liệu identity thực tế trên đĩa (ảnh tham chiếu HOẶC embedding),
        # không dựa vào cờ has_embedding có thể bị reset sai khi cập nhật nhân vật.
        prompt_norm = normalize(scene.image_prompt)
        for char in self.context.characters:
            if (normalize(char.name) in prompt_norm or normalize(char.slug) in prompt_norm) \
                    and self.ctx_mgr.has_identity(char.slug):
                return char.slug

        # Fallback
        if scene.primary_character:
            char = self.ctx_mgr.get_character(scene.primary_character)
            if char and self.ctx_mgr.has_identity(char.slug):
                return char.slug
        return None

    def _resolve_scene_character_any(self, scene) -> Optional[str]:
        """T7: như _resolve_primary_character nhưng KHÔNG yêu cầu identity —
        dùng để phát hiện nhân vật cần bootstrap ref."""
        import unicodedata

        def normalize(s):
            return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower().replace(' ', '').replace('_', '')

        prompt_norm = normalize(scene.image_prompt or "")
        for char in self.context.characters:
            if normalize(char.name) in prompt_norm or normalize(char.slug) in prompt_norm:
                return char.slug
        if scene.primary_character:
            char = self.ctx_mgr.get_character(scene.primary_character)
            if char:
                return char.slug
            # primary_character có thể là TÊN (LLM trả về) thay vì slug
            for char in self.context.characters:
                if normalize(char.name) == normalize(scene.primary_character):
                    return char.slug
        return None

    def _try_bootstrap_ref(self, scene_idx: int, char_slug: str, image) -> bool:
        """T7: lấy ảnh vừa sinh (sau detailer) làm ref cho nhân vật chưa có ảnh.
        Điều kiện: detect được mặt đủ lớn. Trả True nếu bootstrap thành công."""
        try:
            from app.services.storytelling.face_detailer import detect_faces, _expand_box
            faces = detect_faces(image)
            if not faces:
                logger.info(f"[Bootstrap] Cảnh {scene_idx}: không detect được mặt — chưa lấy làm ref.")
                return False
            fx, fy, fw, fh = faces[0]
            if min(fw, fh) < 110:  # draft chưa upscale nên ngưỡng thấp hơn cổng dataset
                logger.info(f"[Bootstrap] Cảnh {scene_idx}: mặt {fw}x{fh}px quá nhỏ — chưa lấy làm ref.")
                return False
            img_w, img_h = image.size
            x0, y0, x1, y1 = _expand_box(fx, fy, fw, fh, img_w, img_h, pad_ratio=0.7)
            face_crop = image.crop((x0, y0, x1, y1))
            if face_crop.size[0] < 512:
                face_crop = face_crop.resize((512, 512))
            ref_path = self.ctx_mgr.set_ref_from_image(char_slug, face_crop, source="auto_bootstrap")
            if ref_path:
                logger.info(f"[Bootstrap] ✅ Nhân vật '{char_slug}' đã có ref tự động từ cảnh {scene_idx} "
                            f"(mặt {fw}x{fh}px): {ref_path}. Các cảnh sau sẽ dùng IP-Adapter.")
                return True
            return False
        except Exception as e:
            logger.warning(f"[Bootstrap] Lỗi bootstrap ref cho '{char_slug}': {e}")
            return False
        
    def _convert_semantic_to_scenes(self, semantic_scenes) -> list:
        """Chuyển SemanticScene (Phase C) → Scene dataclass.
        
        Chọn primary_character theo: nhân vật có identity + xuất hiện đầu tiên.
        Gắn _semantic_meta (attr động, mất sau resume — chấp nhận được).
        """
        import unicodedata
        from app.services.storytelling.models import Scene

        def normalize(s):
            return unicodedata.normalize('NFKD', s).encode(
                'ASCII', 'ignore'
            ).decode('utf-8').lower().replace(' ', '').replace('_', '')

        scenes = []
        for ss in semantic_scenes:
            # Tìm primary_character: nhân vật có identity xuất hiện đầu tiên
            primary = ""
            chars_in_scene = list(ss.characters) if ss.characters else []
            for char_name in chars_in_scene:
                for ctx_char in self.context.characters:
                    if normalize(ctx_char.name) == normalize(char_name):
                        if self.ctx_mgr.has_identity(ctx_char.slug):
                            primary = ctx_char.slug
                            break
                if primary:
                    break

            scene = Scene(
                scene_id=ss.scene_index,
                text_vi=ss.text_vi,
                word_count=len(ss.text_vi.split()),
                start_time=0.0,
                end_time=0.0,
                duration_sec=0.0,
                image_prompt="",
                characters_in_scene=chars_in_scene,
                primary_character=primary,
                fallback_level=0,
                accepted_seed=-1,
                frame_path="",
            )
            # Gắn metadata semantic cho llm_prompter (attr động, mất sau resume)
            scene._semantic_meta = {
                "location": ss.location,
                "action": ss.action,
                "summary": ss.summary_vi,
                "time_of_day": ss.time_of_day,
            }
            scenes.append(scene)
        return scenes

    def step1_generate_script(
        self, 
        md_path: str, 
        audio_path: str, 
        srt_path: str = "", 
        progress_callback: Optional[Callable[[str, int], None]] = None,
        use_whisper: bool = True,
        use_semantic_split: bool = True,
        state_path_override: str = ""
    ) -> Tuple[list, str]:
        """Trạm 1: Đọc kịch bản + chia cảnh + sinh prompt.
        
        Args:
            use_semantic_split: Thử tách cảnh bằng LLM trước (Phase C). Fallback md_parser.
            state_path_override: Path state.json riêng cho batch isolation (FIX Gap4).
        """
        def update_prog(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        # Clear any leftover state from a previous run to avoid orphaned task_dirs
        old_state = self.load_state()
        if old_state and old_state.get("task_dir") and old_state.get("step") != "DONE":
            import shutil
            old_task = old_state["task_dir"]
            if os.path.isdir(old_task):
                try:
                    shutil.rmtree(old_task)
                    logger.info(f"Cleaned up orphaned task dir: {old_task}")
                except Exception as e:
                    logger.warning(f"Could not remove old task dir {old_task}: {e}")
        self.clear_state()

        task_id = str(uuid.uuid4())
        _mc_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        _storage_env = os.getenv("MC_STORAGE_TASKS")
        if _storage_env:
            task_dir = os.path.join(os.path.abspath(_storage_env), "video", "tasks", task_id)
        else:
            task_dir = os.path.join(_mc_root, "storage", "tasks", task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        draft_dir = os.path.join(task_dir, "draft_frames")
        final_dir = os.path.join(task_dir, "final_frames")
        os.makedirs(draft_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        update_prog("1. Đọc Audio và Phân tách kịch bản...", 5)
        # M1: đọc duration CHÍNH XÁC — thất bại thì raise rõ ràng.
        # (Trước đây fallback 60s âm thầm → audio 10 phút bị tách thành 2 cảnh,
        #  video bị cắt còn 60 giây.)
        from app.services.storytelling.audio_utils import get_audio_duration
        total_audio_duration = get_audio_duration(audio_path)
        
        # ---- Nhánh semantic split (Phase C) ----
        semantic_scenes = None
        if use_semantic_split:
            try:
                import codecs
                from app.services.storytelling.semantic_scene_splitter import split_scenes_semantic
                update_prog("1b. Tách cảnh ngữ nghĩa (LLM)...", 8)
                with codecs.open(md_path, "r", "utf-8-sig") as f:
                    md_text = f.read()
                semantic_scenes = split_scenes_semantic(md_text, total_audio_duration)
                if semantic_scenes:
                    logger.info(f"[Step1] Semantic split: {len(semantic_scenes)} cảnh")
                else:
                    logger.info("[Step1] Semantic split trả None → fallback md_parser")
            except Exception as e:
                logger.warning(f"Semantic split error: {e}")
                semantic_scenes = None

        if semantic_scenes is not None:
            # Chuyển SemanticScene → Scene dataclass
            scenes = self._convert_semantic_to_scenes(semantic_scenes)

            # M1 (chính sách 06/07): KHÔNG dùng Whisper.
            # - Có SRT → dùng timing SRT.
            # - Không SRT → map theo tỷ lệ từ trên duration THẬT, rồi TỰ TẠO SRT
            #   từ chính kịch bản (audio đọc nguyên văn) để burn phụ đề.
            update_prog("2. Ghép map timeline SRT...", 15)
            from app.services.storytelling.srt_mapper import (
                parse_srt, map_semantic_scenes_to_srt, generate_srt_from_scenes)
            resolved_srt = srt_path if (srt_path and os.path.exists(srt_path)) else ""
            
            if not resolved_srt:
                update_prog("2a. Dùng Whisper nhận diện thời gian...", 16)
                try:
                    from app.services.subtitle import create_subtitle
                    gen_srt = create_subtitle(audio_path, os.path.join(task_dir, "whisper_sync.srt"), language="vi")
                    if gen_srt and os.path.exists(gen_srt):
                        resolved_srt = gen_srt
                        logger.info(f"[Step1] Đã tạo SRT bằng Whisper: {resolved_srt}")
                except Exception as e:
                    logger.warning(f"Lỗi khi chạy Whisper: {e}")

            blocks = parse_srt(resolved_srt) if resolved_srt else []
            scenes = map_semantic_scenes_to_srt(scenes, blocks, total_audio_duration)

            if not resolved_srt:
                update_prog("2b. Tạo phụ đề từ kịch bản...", 18)
                resolved_srt = generate_srt_from_scenes(
                    scenes, os.path.join(task_dir, "generated_from_script.srt"))
                logger.info(f"[Step1] Đã tạo SRT từ kịch bản: {resolved_srt}")
            srt_path = resolved_srt
        else:
            # ---- Nhánh cũ (md_parser) ----
            scenes = parse_md_to_scenes(md_path, total_audio_duration)
            
            update_prog("2. Ghép map timeline SRT (srt_mapper)...", 15)
            scenes = map_scenes_to_timeline(
                scenes=scenes, 
                srt_path=srt_path, 
                audio_path=audio_path, 
                total_audio_duration=total_audio_duration,
                use_whisper=use_whisper,
                md_path=md_path
            )
            
            if not srt_path or not os.path.exists(srt_path):
                base = os.path.splitext(audio_path)[0]
                possible_srt = base + ".srt"
                if os.path.exists(possible_srt):
                    srt_path = possible_srt
                
        update_prog("3. Sinh prompt LLM (llm_prompter)...", 25)
        scenes = generate_prompts_batch(scenes, self.context)
        
        # FIX Gap4: dùng state_path_override nếu có (batch isolation)
        if state_path_override:
            self._state_path_override = state_path_override
        self.save_state("SCRIPT_READY", scenes, task_dir, audio_path, srt_path, md_path)
        update_prog("Hoàn thành Trạm 1: Kịch bản đã sẵn sàng!", 100)
        return scenes, task_dir

    def step2_generate_images(
        self, 
        scenes: list, 
        task_dir: str, 
        progress_callback: Optional[Callable[[str, int], None]] = None,
        reroll_index: Optional[int] = None,
        new_seed: int = -1,
        new_prompt: str = ""
    ):
        def update_prog(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)
                
        draft_dir = os.path.join(task_dir, "draft_frames")
        os.makedirs(draft_dir, exist_ok=True)
        
        from app.config import load_storytelling_config
        st_config = load_storytelling_config()
        img_w = st_config.get("image_width", 896)
        img_h = st_config.get("image_height", 512)
        
        update_prog("4. Sinh hình ảnh SD (image_generator)... Đang warmup model", 10 if reroll_index is None else 20)
        pipe = StorytellingPipeline(self.context)
        pipe.warmup()
        # T2: ip_adapter_scale đọc từ config — cùng nguồn sự thật với Studio
        if hasattr(pipe, 'update_ip_adapter_scale'):
            pipe.update_ip_adapter_scale(float(st_config.get("ip_adapter_scale", 0.6)))
        
        # ------------------------------------------------------------------
        # T7: Bootstrap ordering — nhân vật CHƯA có ref thì sinh cảnh cận/trung
        # ĐẦU TIÊN của họ trước (mặt to → ref đẹp), các cảnh sau dùng ref đó.
        # Chỉ đổi thứ tự SINH; thứ tự ghép video giữ nguyên theo scene_id.
        # ------------------------------------------------------------------
        if reroll_index is None:
            order = list(range(len(scenes)))
            bootstrap_first = []
            chars_seen = set()
            for i in order:
                slug = self._resolve_scene_character_any(scenes[i])
                if (slug and slug not in chars_seen
                        and not self.ctx_mgr.get_ref_image_path(slug)):
                    chars_seen.add(slug)
                    # tìm cảnh close/medium đầu tiên của nhân vật này
                    best = None
                    for j in order:
                        if self._resolve_scene_character_any(scenes[j]) == slug:
                            if getattr(scenes[j], "shot_type", "wide") in ("close", "medium"):
                                best = j
                                break
                            if best is None:
                                best = j
                    if best is not None and best not in bootstrap_first:
                        bootstrap_first.append(best)
            if bootstrap_first:
                logger.info(f"[Bootstrap] Ưu tiên sinh trước các cảnh {bootstrap_first} "
                            "để lấy ref cho nhân vật chưa có ảnh.")
                indices_to_process = bootstrap_first + [i for i in order if i not in bootstrap_first]
            else:
                indices_to_process = order
        else:
            indices_to_process = [reroll_index]
        total_to_process = max(len(indices_to_process), 1)

        for idx, i in enumerate(indices_to_process):
            scene = scenes[i]
            if reroll_index is not None and new_prompt:
                scene.image_prompt = new_prompt
                
            pct = 20 + int(70 * (idx / total_to_process))
            update_prog(f"4. Sinh hình ảnh SD... Cảnh {i+1}/{len(scenes)}", pct)
            
            face_emb = None
            face_img = None
            primary_slug = self._resolve_primary_character(scene)
            if primary_slug:
                # Ảnh tham chiếu gốc cho IP-Adapter CLIP (hỗ trợ anime)
                ref_path = self.ctx_mgr.get_ref_image_path(primary_slug)
                if ref_path:
                    try:
                        from PIL import Image as PILImage
                        face_img = PILImage.open(ref_path).convert("RGB")
                    except Exception as e:
                        logger.warning(f"Cảnh {i}: không đọc được ảnh tham chiếu của '{primary_slug}': {e}")
                # Embedding cho engine FaceID (nếu dùng)
                import numpy as np
                emb_path = self.ctx_mgr.get_face_embedding_path(primary_slug)
                if os.path.exists(emb_path):
                    try:
                        face_emb = np.load(emb_path)
                    except Exception as e:
                        logger.warning(f"Cảnh {i}: không đọc được face embedding của '{primary_slug}': {e}")
                if face_img is None and face_emb is None:
                    logger.warning(f"Cảnh {i}: nhân vật '{primary_slug}' không có dữ liệu identity. "
                                   "Ảnh sẽ sinh KHÔNG có điều kiện khuôn mặt.")

            # T7: nhân vật của cảnh này chưa có ref? → ứng viên bootstrap
            bootstrap_slug = None
            if primary_slug is None:
                any_slug = self._resolve_scene_character_any(scene)
                if any_slug and not self.ctx_mgr.get_ref_image_path(any_slug):
                    bootstrap_slug = any_slug
            # LoRA nhân vật (nếu đã train) — giữ identity mạnh hơn IP-Adapter
            if hasattr(pipe, 'set_character_lora'):
                pipe.set_character_lora(primary_slug)

            # T5: chọn khung sinh theo cỡ cảnh — cảnh cận sinh khung dọc để mặt
            # to ngay từ draft (SD1.5 vẽ mặt nhỏ rất tệ). Video assembler sẽ
            # pad về 16:9 (force_original_aspect_ratio=decrease,pad) nên không vỡ khung.
            shot_map = {
                "close":  (576, 704),   # dọc — mặt chiếm khung lớn
                "medium": (704, 528),   # 4:3 ngang
                "wide":   (img_w, img_h),  # 16:9 theo config
            }
            scene_w, scene_h = shot_map.get(getattr(scene, "shot_type", "wide"), (img_w, img_h))
            if (scene_w, scene_h) != (img_w, img_h):
                logger.info(f"Cảnh {i}: shot_type={scene.shot_type} → sinh khung {scene_w}x{scene_h}")

            draft_img, seed = pipe.generate_draft(
                prompt=scene.image_prompt,
                negative_prompt=self.context.get_negative_prompt(),
                face_embedding=face_emb,
                face_image=face_img,
                seed=new_seed if reroll_index is not None else -1,
                width=scene_w,
                height=scene_h
            )

            # Face Detailer: vẽ lại mặt ngay khi SD còn trên VRAM (tự động cho batch,
            # chạy TRƯỚC auto-collect để dataset chỉ nhận mặt đã sửa)
            if st_config.get("enable_face_detailer", True):
                from app.services.storytelling.face_detailer import detail_faces
                draft_img = detail_faces(
                    pipe, draft_img,
                    prompt=scene.image_prompt,
                    negative_prompt=self.context.get_negative_prompt(),
                    face_image=face_img, face_embedding=face_emb,
                    strength=st_config.get("face_detailer_strength", 0.45),
                )

            # T7: bootstrap ref — ảnh đạt chuẩn đầu tiên của nhân vật chưa có ảnh
            # (chạy SAU face detailer để ref là mặt đã được sửa)
            if bootstrap_slug:
                self._try_bootstrap_ref(i, bootstrap_slug, draft_img)

            frame_path = os.path.join(draft_dir, f"scene_{i:03d}.png")
            draft_img.save(frame_path)

            # T6: KHÔNG thu thập dataset từ ảnh draft — chỉ ĐO TRƯỚC similarity
            # (CLIP encoder còn trên VRAM lúc này; step3 upscale chạy sau khi SD
            # release nên không đo được nữa). Thu thập thật diễn ra ở step3.
            scene._collect_slug = ""
            scene._collect_sim = None
            if primary_slug and len(scene.characters_in_scene) == 1:
                try:
                    from app.services.storytelling.dataset_collector import precheck_similarity
                    scene._collect_slug = primary_slug
                    scene._collect_sim = precheck_similarity(self.ctx_mgr, primary_slug, draft_img)
                except Exception as e_pre:
                    logger.debug(f"Precheck similarity skip: {e_pre}")

            scene.accepted_seed = seed
            scene.frame_path = frame_path
            
        # Giải phóng rác bộ nhớ CUDA tạm thời mà không hủy mô hình SD nếu không cần thiết
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        state = self.load_state() or {}
        audio_path = state.get("audio_path", "")
        srt_path = state.get("srt_path", "")
        md_path = state.get("md_path", "")
        
        self.save_state("STORYBOARD_READY", scenes, task_dir, audio_path, srt_path, md_path)
        update_prog("Hoàn thành Trạm 2: Storyboard đã sẵn sàng!", 100)
        return scenes

    def reroll_scene(self, scene_index: int, new_seed: int = -1, new_prompt: str = "") -> str:
        state = self.load_state()
        if not state or "scenes" not in state or not state.get("task_dir"):
            raise ValueError("No active state found to reroll scene.")
        from app.services.storytelling.models import scene_from_dict
        scenes = [scene_from_dict(s) for s in state["scenes"]]
        task_dir = state["task_dir"]
        
        self.step2_generate_images(scenes, task_dir, reroll_index=scene_index, new_seed=new_seed, new_prompt=new_prompt)
        return scenes[scene_index].frame_path

    def step3_render_final(
        self, 
        scenes: list, 
        task_dir: str, 
        audio_path: str, 
        srt_path: str, 
        bgm_path: str = "", 
        bgm_volume: float = 0.15,
        enable_upscaling: bool = True,
        burn_subtitles: bool = True,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> str:
        def update_prog(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)
                
        final_dir = os.path.join(task_dir, "final_frames")
        os.makedirs(final_dir, exist_ok=True)

        # NOTE: enable_upscaling is a per-render runtime flag.
        # We do NOT write it to config.toml to avoid mutating the global config file.
        if enable_upscaling:
            # Chỉ khi cần upscale RealESRGAN mới giải phóng Stable Diffusion để nhường trọn vẹn 8GB VRAM
            try:
                from app.services.storytelling.image_generator import StorytellingPipeline
                StorytellingPipeline().release()
            except Exception as e:
                logger.warning(f"Could not release SD pipeline: {e}")

        from app.services.storytelling.hardware_adapter import get_hardware_config
        hw_config = get_hardware_config()
        update_prog("5. Hậu kỳ (postprocess)...", 10)
        postproc = PostProcessor(device=hw_config["esrgan_device"], enable_upscaling=enable_upscaling)
        frame_paths = [s.frame_path for s in scenes]
        
        def post_cb(curr, tot):
            pct = 10 + int(60 * (curr / max(tot, 1)))
            update_prog(f"5. Hậu kỳ... Cảnh {curr}/{tot}", pct)
            
        final_paths = postproc.process_all(frame_paths, final_dir, pipeline=None, on_progress=post_cb)

        for i, scene in enumerate(scenes):
            scene.frame_path = final_paths[i]

            # T6: thu thập dataset SAU upscale — cổng chất lượng (mặt ≥160px +
            # similarity đã đo trước ở step2) nằm trong maybe_collect.
            collect_slug = getattr(scene, "_collect_slug", "")
            if collect_slug:
                try:
                    from PIL import Image as PILImage
                    from app.services.storytelling.dataset_collector import maybe_collect
                    upscaled_img = PILImage.open(final_paths[i]).convert("RGB")
                    maybe_collect(self.ctx_mgr, collect_slug, upscaled_img, "auto",
                                  precomputed_sim=getattr(scene, "_collect_sim", None))
                except Exception as e_collect:
                    logger.debug(f"Dataset collect skip (scene {i}): {e_collect}")
            
        update_prog("6. Lắp ráp video (video_assembler)...", 80)
        final_video_path = os.path.join(task_dir, "final_video.mp4")
        
        frames_info = [FrameInfo(frame_path=s.frame_path, duration_sec=s.duration_sec) for s in scenes]
        
        assemble_video(
            frames=frames_info,
            audio_path=audio_path,
            srt_path=srt_path,
            output_path=final_video_path,
            burn_subtitles=burn_subtitles,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume
        )
        
        state = self.load_state() or {}
        md_path = state.get("md_path", "")
        self.save_state("DONE", scenes, task_dir, audio_path, srt_path, md_path)
        update_prog("Hoàn thành Trạm 3: Video Final đã sẵn sàng!", 100)
        return final_video_path
        
    def run_pipeline(
        self, 
        md_path: str, 
        audio_path: str, 
        srt_path: str, 
        progress_callback: Optional[Callable[[str, int], None]] = None,
        enable_upscaling: bool = True
    ) -> str:
        """
        Chạy tự động toàn bộ pipeline từ đầu đến cuối (Skip checkpoints) và trả về path video final.
        """
        scenes, task_dir = self.step1_generate_script(md_path, audio_path, srt_path, progress_callback, use_whisper=False)
        scenes = self.step2_generate_images(scenes, task_dir, progress_callback)
        # Reload srt_path in case Whisper generated it during step1
        state = self.load_state() or {}
        resolved_srt = state.get("srt_path", srt_path)
        
        # Chế độ chạy tự động nhanh (Skip Checkpoints): 
        # Tắt phụ đề (burn_subtitles = False) nếu người dùng không chủ động up file SRT để tối ưu thời gian.
        should_burn = bool(srt_path and os.path.exists(srt_path))

        final_video_path = self.step3_render_final(
            scenes, task_dir, audio_path, resolved_srt, 
            burn_subtitles=should_burn,
            enable_upscaling=enable_upscaling,
            progress_callback=progress_callback
        )
        return final_video_path
