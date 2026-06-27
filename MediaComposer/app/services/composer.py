import os
from loguru import logger
from urllib.parse import urlparse
from app.services.video import generate_video, combine_videos
from app.services.subtitle import create_subtitle, create_subtitle_from_text, read_srt_text
from app.services.llm import extract_search_terms
from app.services.material import download_videos
from app.utils import utils
from app.config import config
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
        threads: int = None,
        transcript_text: str = ""
    ) -> str:
        """
        Orchestrates the entire flow:
        1. Subtitle generation
        2. LLM keyword extraction & Video fetching (if auto_fetch)
        3. Video combination (with loop)
        4. Final composition
        """
        task_dir = utils.task_dir(task_id)
        
        # 1. Subtitle generation / transcript handling
        subtitle_path = ""
        srt_text = ""
        if transcript_text.strip():
            # Fast path: user provided transcript text, skip Whisper entirely
            logger.info("⚡ Transcript text provided — skipping Whisper transcription")
            srt_text = transcript_text
            if enable_subtitles:
                subtitle_path = os.path.join(task_dir, "subtitle.srt")
                audio_dur = self._get_audio_duration(audio_path)
                create_subtitle_from_text(transcript_text, audio_dur, subtitle_path)
        elif enable_subtitles or auto_fetch:
            # Original path: use Whisper for transcription
            logger.info("Generating subtitles from audio (Whisper)...")
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
        
        # Determine background color based on background style
        bg_style = config.whisper.get("background_style", "None")
        if bg_style == "None":
            bg_color = False
        elif bg_style == "Black":
            bg_color = "#000000"
        else:
            bg_color = config.whisper.get("text_background_color", "#000000")

        params = VideoParams(
            video_aspect=video_aspect,
            video_concat_mode=concat_mode,
            voice_volume=1.0,
            bgm_type="custom" if bgm_file else "",
            bgm_file=bgm_file,
            bgm_volume=0.2 if bgm_file else 0.0,
            subtitle_enabled=enable_subtitles,
            font_name=config.whisper.get("font_name", "STHeitiMedium.ttc"),
            font_size=int(config.whisper.get("font_size", 60)),
            text_fore_color=config.whisper.get("text_fore_color", "#FFFFFF"),
            stroke_color=config.whisper.get("stroke_color", "#000000"),
            stroke_width=float(config.whisper.get("stroke_width", 1.5)),
            text_background_color=bg_color,
            subtitle_bg_alpha=int(config.whisper.get("subtitle_bg_alpha", 140)),
            rounded_subtitle_background=bool(config.whisper.get("rounded_subtitle_background", False)),
            subtitle_position=config.whisper.get("subtitle_position", "bottom"),
            custom_position=float(config.whisper.get("custom_position", 70.0)),
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
