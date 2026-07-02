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
        
    def step1_generate_script(
        self, 
        md_path: str, 
        audio_path: str, 
        srt_path: str = "", 
        progress_callback: Optional[Callable[[str, int], None]] = None,
        use_whisper: bool = True
    ) -> Tuple[list, str]:
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
        task_dir = os.path.join("storage", "tasks", task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        draft_dir = os.path.join(task_dir, "draft_frames")
        final_dir = os.path.join(task_dir, "final_frames")
        os.makedirs(draft_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        update_prog("1. Đọc Audio và Phân tách kịch bản (md_parser)...", 5)
        total_audio_duration = 60.0 
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            total_audio_duration = len(audio) / 1000.0
        except Exception as e:
            logger.warning(f"Could not read audio duration (pydub error): {e}")
            
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
        
        indices_to_process = range(len(scenes)) if reroll_index is None else [reroll_index]
        total_to_process = max(len(indices_to_process), 1)
        
        for idx, i in enumerate(indices_to_process):
            scene = scenes[i]
            if reroll_index is not None and new_prompt:
                scene.image_prompt = new_prompt
                
            pct = 20 + int(70 * (idx / total_to_process))
            update_prog(f"4. Sinh hình ảnh SD... Cảnh {i+1}/{len(scenes)}", pct)
            
            face_emb = None
            if scene.primary_character:
                char = self.ctx_mgr.get_character(scene.primary_character)
                if char and char.has_embedding:
                    import numpy as np
                    try:
                        emb_path = os.path.join(self.ctx_mgr.chars_dir, char.slug, "face.ipadpt")
                        face_emb = np.load(emb_path)
                    except:
                        pass
                        
            draft_img, seed = pipe.generate_draft(
                prompt=scene.image_prompt,
                negative_prompt=self.context.get_negative_prompt(),
                face_embedding=face_emb,
                seed=new_seed if reroll_index is not None else -1,
                width=img_w,
                height=img_h
            )
            
            frame_path = os.path.join(draft_dir, f"scene_{i:03d}.png")
            draft_img.save(frame_path)
            
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
        from app.services.storytelling.models import Scene
        scenes = [Scene(**s) for s in state["scenes"]]
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
