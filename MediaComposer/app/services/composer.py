import os
from loguru import logger
from urllib.parse import urlparse
from app.services.video import generate_video, combine_videos
from app.services.subtitle import create_subtitle, read_srt_text
from app.services.llm import extract_search_terms
from app.services.material import download_videos
from app.utils import utils
from app.models.schema import VideoAspect, VideoConcatMode, VideoTransitionMode, VideoParams

class ComposerWorkflow:
    def __init__(self):
        pass
        
    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            clip = AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
            return duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 60.0

    def run_workflow(
        self,
        task_id: str,
        audio_path: str,
        video_paths: list[str] = None, # Used for manual mode
        auto_fetch: bool = False,
        source: str = "pexels",
        bgm_file: str = "",
        video_aspect: VideoAspect = VideoAspect.portrait,
        concat_mode: VideoConcatMode = VideoConcatMode.random,
        enable_subtitles: bool = True,
        max_clip_duration: int = 5,
        threads: int = None
    ) -> str:
        """
        Orchestrates the entire flow:
        1. Subtitle generation
        2. LLM keyword extraction & Video fetching (if auto_fetch)
        3. Video combination (with loop)
        4. Final composition
        """
        task_dir = utils.task_dir(task_id)
        
        # 1. Subtitle generation
        subtitle_path = ""
        srt_text = ""
        if enable_subtitles or auto_fetch:
            logger.info("Generating subtitles from audio...")
            subtitle_path = os.path.join(task_dir, "subtitle.srt")
            create_subtitle(audio_path, subtitle_path)
            srt_text = read_srt_text(subtitle_path)
            if not enable_subtitles:
                subtitle_path = "" # We just needed the text for LLM, don't burn it in

        # 2. Materials
        materials_to_use = video_paths or []
        audio_duration = self._get_audio_duration(audio_path)
        
        if auto_fetch:
            logger.info("Extracting search terms using LLM...")
            search_terms = extract_search_terms(srt_text, amount=5)
            if not search_terms:
                search_terms = ["nature"] # fallback
            logger.info(f"Search terms: {search_terms}")
            
            # Fetch videos
            fetched_paths = download_videos(
                task_id=task_id,
                search_terms=search_terms,
                source=source,
                video_aspect=video_aspect,
                video_concat_mode=concat_mode,
                audio_duration=audio_duration,
                max_clip_duration=max_clip_duration,
                match_script_order=False
            )
            materials_to_use.extend(fetched_paths)

        if not materials_to_use:
            raise ValueError("No video materials available to compose.")

        # 3. Combine Videos
        merged_video_path = os.path.join(task_dir, "merged.mp4")
        logger.info(f"Combining {len(materials_to_use)} videos...")
        combine_videos(
            combined_video_path=merged_video_path,
            video_paths=materials_to_use,
            audio_file=audio_path,
            video_aspect=video_aspect,
            video_concat_mode=concat_mode,
            video_transition_mode=None,
            max_clip_duration=max_clip_duration,
            threads=threads
        )

        # 4. Final Compose
        final_video_path = os.path.join(task_dir, "final.mp4")
        params = VideoParams(
            video_aspect=video_aspect,
            video_concat_mode=concat_mode,
            voice_volume=1.0,
            bgm_type="custom" if bgm_file else "",
            bgm_file=bgm_file,
            bgm_volume=0.2 if bgm_file else 0.0,
            subtitle_enabled=enable_subtitles,
            n_threads=threads
        )
        
        logger.info("Generating final video...")
        generate_video(
            video_path=merged_video_path,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params
        )
        
        return final_video_path

composer = ComposerWorkflow()
