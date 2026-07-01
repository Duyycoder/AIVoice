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
from typing import Callable, Optional

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
        
    def run_pipeline(
        self, 
        md_path: str, 
        audio_path: str, 
        srt_path: str, 
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> str:
        """
        Chạy toàn bộ pipeline từ đầu đến cuối và trả về path của video nháp.
        """
        def update_prog(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)
                
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
            total_audio_duration=total_audio_duration
        )
        
        # Nếu srt_path trống, srt_mapper đã gọi Whisper tạo file mới. Cần tìm file đó để burn sub.
        if not srt_path or not os.path.exists(srt_path):
            base = os.path.splitext(audio_path)[0]
            possible_srt = base + ".srt"
            if os.path.exists(possible_srt):
                srt_path = possible_srt
                
        update_prog("3. Sinh prompt LLM (llm_prompter)...", 25)
        scenes = generate_prompts_batch(scenes, self.context)
        
        update_prog("4. Sinh hình ảnh SD (image_generator)... Đang warmup model", 35)
        pipe = StorytellingPipeline(self.context)
        pipe.warmup()
        
        for i, scene in enumerate(scenes):
            update_prog(f"4. Sinh hình ảnh SD... Cảnh {i+1}/{len(scenes)}", 35 + int(35 * (i/len(scenes))))
            
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
                face_embedding=face_emb
            )
            
            frame_path = os.path.join(draft_dir, f"scene_{i:03d}.png")
            draft_img.save(frame_path)
            
            scene.accepted_seed = seed
            scene.frame_path = frame_path
            
        pipe.release()
        
        update_prog("5. Hậu kỳ (postprocess)...", 75)
        postproc = PostProcessor()
        frame_paths = [s.frame_path for s in scenes]
        final_paths = postproc.process_all(frame_paths, final_dir, pipeline=None)
        
        for i, scene in enumerate(scenes):
            scene.frame_path = final_paths[i]
            
        update_prog("6. Lắp ráp video (video_assembler)...", 90)
        draft_video_path = os.path.join(task_dir, "draft_video.mp4")
        
        frames_info = [FrameInfo(frame_path=s.frame_path, duration_sec=s.duration_sec) for s in scenes]
        
        assemble_video(
            frames=frames_info,
            audio_path=audio_path,
            srt_path=srt_path,
            output_path=draft_video_path,
            burn_subtitles=True
        )
        
        update_prog("Hoàn thành Pipeline!", 100)
        return draft_video_path
