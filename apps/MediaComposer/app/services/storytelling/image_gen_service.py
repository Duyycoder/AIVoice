import os
import json
import uuid
from typing import List, Callable, Optional
from loguru import logger

from app.services.storytelling.context_manager import ContextManager
from app.services.storytelling.models import ImageGenParams, ImageGenTask, ImageGenResult
from app.services.storytelling.prompt_translator import PromptTranslator
from app.services.storytelling.image_generator import StorytellingPipeline
from app.services.storytelling.postprocess import PostProcessor

class ImageGenOrchestrator:
    def __init__(self, ctx_mgr: ContextManager):
        self.ctx_mgr = ctx_mgr
        self.translator = PromptTranslator(ctx_mgr)
        self._state_file = os.path.join(ctx_mgr.context_dir, "image_gen_state.json")
        self.pipeline = StorytellingPipeline(ctx_mgr.load_context())
        self.postprocessor = PostProcessor()

    def step1_prepare_prompts(
        self,
        raw_descriptions: List[str],
        style_preset: str = "",
        custom_style: str = "",
        is_vietnamese: bool = True,
        progress_callback: Callable = None
    ) -> List[ImageGenTask]:
        
        tasks = []
        total = len(raw_descriptions)
        for i, desc in enumerate(raw_descriptions):
            if not desc.strip():
                continue
                
            pos_prompt, neg_prompt, char_slugs, primary_slug = self.translator.translate_and_build_prompt(
                desc, style_preset, custom_style, is_vietnamese
            )
            
            task = ImageGenTask(
                task_id=i,
                original_description=desc,
                processed_prompt=pos_prompt,
                negative_prompt=neg_prompt,
                character_slugs=char_slugs,
                primary_character=primary_slug,
                style_preset=style_preset or "custom"
            )
            tasks.append(task)
            
            if progress_callback:
                progress_callback(i + 1, total)
                
        return tasks

    def _get_face_embedding(self, char_slug: str):
        if not char_slug:
            return None
        import numpy as np
        emb_path = self.ctx_mgr.get_face_embedding_path(char_slug)
        if os.path.exists(emb_path):
            try:
                return np.load(emb_path)
            except Exception as e:
                logger.error(f"Error loading face embedding for {char_slug}: {e}")
        else:
            logger.warning(f"Nhân vật '{char_slug}' chưa có face embedding ({emb_path}). "
                           "Hãy upload ảnh tham chiếu và bấm trích xuất khuôn mặt trong Context Window.")
        return None

    def _get_face_image(self, char_slug: str):
        """Ảnh tham chiếu gốc của nhân vật (PIL) — dùng cho IP-Adapter CLIP (hỗ trợ anime)."""
        if not char_slug:
            return None
        ref_path = self.ctx_mgr.get_ref_image_path(char_slug)
        if ref_path:
            try:
                from PIL import Image
                return Image.open(ref_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading ref image for {char_slug}: {e}")
        else:
            logger.warning(f"Nhân vật '{char_slug}' chưa có ảnh tham chiếu (ref.*) trong Context Window.")
        return None

    def get_pipeline_warnings(self) -> List[str]:
        """Cảnh báo từ pipeline (VD: IP-Adapter load fail) để UI hiển thị."""
        return list(getattr(self.pipeline, "warnings", []))

    def step2_generate_images(
        self,
        tasks: List[ImageGenTask],
        gen_params: ImageGenParams,
        progress_callback: Callable = None
    ) -> List[ImageGenResult]:
        
        results = []
        total = len(tasks)
        
        # Temp dir for drafts
        draft_dir = os.path.join(self.ctx_mgr.context_dir, "image_gen_drafts")
        os.makedirs(draft_dir, exist_ok=True)
        
        # Init pipeline với tham số THỰC TẾ (scheduler/LoRA khớp chế độ sinh ảnh)
        self.pipeline.warmup(
            num_steps=gen_params.num_inference_steps,
            guidance_scale=gen_params.guidance_scale
        )

        if hasattr(self.pipeline, 'update_ip_adapter_scale'):
            self.pipeline.update_ip_adapter_scale(gen_params.ip_adapter_scale)
            
        for i, task in enumerate(tasks):
            face_emb = self._get_face_embedding(task.primary_character)
            face_img = self._get_face_image(task.primary_character)
            # LoRA nhân vật (nếu đã train) — giữ identity mạnh hơn IP-Adapter
            if hasattr(self.pipeline, 'set_character_lora'):
                self.pipeline.set_character_lora(task.primary_character)

            seed = gen_params.seed
            if seed == -1:
                import random
                seed = random.randint(1, 999999999)
                
            out_filename = f"task_{task.task_id}_{uuid.uuid4().hex[:6]}.png"
            out_path = os.path.join(draft_dir, out_filename)
            
            try:
                draft_img, final_seed = self.pipeline.generate_draft(
                    prompt=task.processed_prompt,
                    negative_prompt=task.negative_prompt,
                    face_embedding=face_emb,
                    face_image=face_img,
                    seed=seed,
                    width=gen_params.width,
                    height=gen_params.height,
                    num_steps=gen_params.num_inference_steps,
                    guidance_scale=gen_params.guidance_scale
                )
                # Face Detailer: vẽ lại mặt ngay khi SD còn trên VRAM (tự động, cả batch)
                if gen_params.enable_face_detailer:
                    from app.services.storytelling.face_detailer import detail_faces
                    from app.config import load_storytelling_config as _lsc
                    draft_img = detail_faces(
                        self.pipeline, draft_img,
                        prompt=task.processed_prompt,
                        negative_prompt=task.negative_prompt,
                        face_image=face_img, face_embedding=face_emb,
                        num_steps=gen_params.num_inference_steps,
                        guidance_scale=gen_params.guidance_scale,
                        strength=_lsc().get("face_detailer_strength", 0.45),
                    )
                draft_img.save(out_path)
                success = True
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                success = False
            
            if success:
                res = ImageGenResult(
                    task_id=task.task_id,
                    image_path=out_path,
                    seed=final_seed,
                    prompt_used=task.processed_prompt,
                    character_slugs=task.character_slugs
                )
                results.append(res)
                # T6: KHÔNG thu thập dataset ở Trạm 2 nữa (ảnh draft mặt nhỏ dễ
                # nhiễm dữ liệu train) — hook chuyển sang step3_upscale_export.

            if progress_callback:
                progress_callback(i + 1, total)
                
        # FIX: KHÔNG release() toàn bộ pipeline sau mỗi batch (trước đây khiến
        # lần bấm kế tiếp phải load lại model từ disk ~30-60s).
        # Chỉ dọn rác VRAM tạm; model được giữ lại cho batch tiếp theo.
        self.pipeline.free_vram_cache()

        return results

    def reroll_image(
        self,
        task: ImageGenTask,
        gen_params: ImageGenParams,
        new_prompt: str = ""
    ) -> Optional[ImageGenResult]:
        
        draft_dir = os.path.join(self.ctx_mgr.context_dir, "image_gen_drafts")
        os.makedirs(draft_dir, exist_ok=True)
        
        self.pipeline.warmup(
            num_steps=gen_params.num_inference_steps,
            guidance_scale=gen_params.guidance_scale
        )
        if hasattr(self.pipeline, 'update_ip_adapter_scale'):
            self.pipeline.update_ip_adapter_scale(gen_params.ip_adapter_scale)

        face_emb = self._get_face_embedding(task.primary_character)
        face_img = self._get_face_image(task.primary_character)
        if hasattr(self.pipeline, 'set_character_lora'):
            self.pipeline.set_character_lora(task.primary_character)

        seed = gen_params.seed
        if seed == -1:
            import random
            seed = random.randint(1, 999999999)

        out_filename = f"task_{task.task_id}_reroll_{uuid.uuid4().hex[:6]}.png"
        out_path = os.path.join(draft_dir, out_filename)
        
        prompt = new_prompt if new_prompt else task.processed_prompt
        
        try:
            draft_img, final_seed = self.pipeline.generate_draft(
                prompt=prompt,
                negative_prompt=task.negative_prompt,
                face_embedding=face_emb,
                face_image=face_img,
                seed=seed,
                width=gen_params.width,
                height=gen_params.height,
                num_steps=gen_params.num_inference_steps,
                guidance_scale=gen_params.guidance_scale
            )
            if gen_params.enable_face_detailer:
                from app.services.storytelling.face_detailer import detail_faces
                from app.config import load_storytelling_config as _lsc
                draft_img = detail_faces(
                    self.pipeline, draft_img,
                    prompt=prompt,
                    negative_prompt=task.negative_prompt,
                    face_image=face_img, face_embedding=face_emb,
                    num_steps=gen_params.num_inference_steps,
                    guidance_scale=gen_params.guidance_scale,
                    strength=_lsc().get("face_detailer_strength", 0.45),
                )
            draft_img.save(out_path)
            success = True
        except Exception as e:
            logger.error(f"Image reroll failed: {e}")
            success = False
        
        if success:
            return ImageGenResult(
                task_id=task.task_id,
                image_path=out_path,
                seed=final_seed,
                prompt_used=prompt,
                character_slugs=task.character_slugs
            )
        return None

    def step3_upscale_export(
        self,
        results: List[ImageGenResult],
        enable_upscale: bool = True,
        target_quality: str = "2k",
        output_dir: str = "",
        batch_folder_name: str = "",
        progress_callback: Callable = None
    ) -> List[str]:
        
        if not output_dir:
            output_dir = os.path.join(self.ctx_mgr.context_dir, "generated_images")
            
        if batch_folder_name:
            final_dir = os.path.join(output_dir, batch_folder_name)
        else:
            final_dir = output_dir
            
        os.makedirs(final_dir, exist_ok=True)
        
        # FIX: target theo ASPECT RATIO của ảnh gốc (trước đây ép cứng 16:9 →
        # ảnh dọc 9:16 bị crop nát khi xuất 1920x1080).
        # Chỉ chốt CẠNH DÀI theo chất lượng; cạnh ngắn tính theo tỷ lệ ảnh.
        long_side = {"1k": 1280, "2k": 1920, "4k": 3840}.get(target_quality, 1920)

        final_paths = []
        total = len(results)
        from PIL import Image
        
        # Temporarily override upscale setting
        self.postprocessor._enable_upscaling_override = enable_upscale
        
        for i, res in enumerate(results):
            if not os.path.exists(res.image_path):
                continue
                
            img = Image.open(res.image_path).convert("RGB")

            # Tính target giữ nguyên aspect ratio ảnh gốc (làm tròn bội số 2)
            w, h = img.size
            if w >= h:
                target_w = long_side
                target_h = max(2, int(round(long_side * h / w / 2)) * 2)
            else:
                target_h = long_side
                target_w = max(2, int(round(long_side * w / h / 2)) * 2)

            # Upscale
            upscaled = self.postprocessor.run_realesrgan(
                img,
                scale=4,
                target_w=target_w,
                target_h=target_h
            )
            
            out_name = f"gen_{res.task_id}_{uuid.uuid4().hex[:6]}.png"
            out_path = os.path.join(final_dir, out_name)

            upscaled.save(out_path, format="PNG")

            # T6: thu thập dataset SAU upscale (user đã Chấp nhận mới tới Trạm 3).
            # Cổng chất lượng (mặt ≥160px + CLIP similarity) nằm trong maybe_collect.
            if len(res.character_slugs) == 1 and res.character_slugs[0]:
                try:
                    from app.services.storytelling.dataset_collector import maybe_collect
                    maybe_collect(self.ctx_mgr, res.character_slugs[0], upscaled, "approved")
                except Exception as e_collect:
                    logger.debug(f"Dataset collect skip: {e_collect}")
            final_paths.append(out_path)
            
            if progress_callback:
                progress_callback(i + 1, total)
                
        # clear upscaler VRAM
        from app.services.storytelling.postprocess import free_realesrgan_cache
        free_realesrgan_cache()
        
        return final_paths
