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
        transcript_text: str = "",
        slice_video: bool = True
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
                search_terms = ["nature"]  # fallback
            # Add extra generic terms for richer video pool (Option 4)
            extra_terms = ["fantasy", "magic", "battle", "adventure"]
            search_terms.extend(extra_terms)
            logger.info(f"Search terms (including extras): {search_terms}")

            # Fetch videos (Option 2: increase max_clip_duration, Option 5: increase threads)
            fetched_paths = download_videos(
                task_id=task_id,
                search_terms=search_terms,
                source=source,
                video_aspect=video_aspect,
                video_concat_mode=concat_mode,
                audio_duration=audio_duration,
                max_clip_duration=30,  # increased from default 5 seconds
                match_script_order=False,
                threads=os.cpu_count() or 4,  # increased threads (Option 5)
            )
            materials_to_use.extend(fetched_paths)

        if not materials_to_use:
            raise ValueError("No video materials available to compose.")

        # Limit total video duration to 30 minutes (Option 3) only for Auto Fetch mode
        if auto_fetch:
            MAX_TOTAL_VIDEO_SECONDS = 30 * 60
            total_video_seconds = 0.0
            limited_materials = []
            from moviepy.video.io.VideoFileClip import VideoFileClip
            for vp in materials_to_use:
                try:
                    clip = VideoFileClip(vp)
                    dur = clip.duration
                    clip.close()
                    if total_video_seconds + dur > MAX_TOTAL_VIDEO_SECONDS:
                        break
                    total_video_seconds += dur
                    limited_materials.append(vp)
                except Exception:
                    continue
            if limited_materials:
                materials_to_use = limited_materials

        # 3. Combine Videos
        merged_video_path = os.path.join(task_dir, "merged.mp4")
        
        is_single_video = len(materials_to_use) == 1
        bypass_combine = False
        
        if is_single_video:
            single_video_path = materials_to_use[0]
            # Get single video duration
            from moviepy.video.io.VideoFileClip import VideoFileClip
            try:
                temp_clip = VideoFileClip(single_video_path)
                video_duration = temp_clip.duration
                temp_clip.close()
            except Exception as e:
                logger.error(f"Failed to get video duration: {e}")
                video_duration = 0.0
                
            # If no subtitles and no BGM, and video is long enough, do a Fast Stream Copy
            if not enable_subtitles and not bgm_file and video_duration >= audio_duration:
                logger.info("⚡ Fast Path: Subtitles and BGM disabled, video >= audio. Merging directly via FFmpeg stream copy...")
                import subprocess
                ffmpeg_bin = utils.get_ffmpeg_binary()
                final_video_path = os.path.join(task_dir, "final.mp4")
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-ss", "0",
                    "-t", f"{audio_duration:.3f}",
                    "-i", single_video_path,
                    "-i", audio_path,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    final_video_path
                ]
                logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    logger.success("⚡ Fast merge completed successfully!")
                    return final_video_path
                else:
                    logger.warning(f"Fast merge failed (code {res.returncode}): {res.stderr}. Falling back to standard flow.")
            
            # Otherwise, we can bypass combine_videos, set merged_video_path to the single video directly,
            # and let generate_video handle virtual resize and virtual trim/loop.
            logger.info("⚡ Single video detected. Bypassing combine_videos to avoid re-encoding twice.")
            merged_video_path = single_video_path
            bypass_combine = True

        if not bypass_combine:
            logger.info(f"Combining {len(materials_to_use)} videos...")
            combine_videos(
                combined_video_path=merged_video_path,
                video_paths=materials_to_use,
                audio_file=audio_path,
                video_aspect=video_aspect,
                video_concat_mode=concat_mode,
                video_transition_mode=None,
                max_clip_duration=30,  # ensure consistency
                threads=threads or os.cpu_count() or 4,
                slice_video=slice_video,
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

    def split_video_into_parts(
        self,
        task_id: str,
        video_path: str,
        num_parts: int,
        fast_split: bool = True
    ) -> list[str]:
        task_dir = utils.task_dir(task_id)
        from app.services.video import split_video_file
        
        output_files = split_video_file(
            video_path=video_path,
            num_parts=num_parts,
            output_dir=task_dir,
            fast_split=fast_split
        )
        
        # Save manifest
        manifest_path = os.path.join(task_dir, "split_manifest.json")
        import json
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "split",
                "parts": [os.path.basename(p) for p in output_files]
            }, f, indent=2)
            
        return output_files

    def run_translation_workflow(
        self,
        task_id: str,
        video_path: str,
        source_lang: str,
        burn_method: str = "ffmpeg",
        enable_voiceover: bool = False,
        tts_engine: str = "edge",
        tts_voice: str = "",
        ducking_ratio: float = 90.0,
        auto_clone: bool = False
    ) -> str:
        """
        Orchestrates the automatic translation and subtitling workflow:
        1. Extract audio from video.
        2. Transcribe audio to source SRT (using Whisper) in the source language.
        3. Release Whisper model to save VRAM.
        4. Translate SRT to Vietnamese using Gemini.
        5. (Optional) Generate Vietnamese voiceover and apply Dynamic Audio Ducking.
        6. Burn subtitles into the video (via FFmpeg native filter or MoviePy) using the mixed audio.
        """
        import subprocess
        from app.services.translation import translate_srt
        from app.services.video import burn_subtitles_ffmpeg, generate_video
        from app.services.subtitle import release_whisper_model
        
        task_dir = utils.task_dir(task_id)
        logger.info(f"Starting automatic translation workflow for video: {video_path}")
        
        # 1. Extract audio from video
        audio_path = os.path.join(task_dir, "extracted_audio.wav")
        ffmpeg_bin = utils.get_ffmpeg_binary()
        
        # We convert to mono, 16kHz PCM WAV for Whisper optimization
        audio_cmd = [
            ffmpeg_bin,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            audio_path
        ]
        logger.info(f"Extracting optimized audio track: {' '.join(audio_cmd)}")
        res = subprocess.run(audio_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"Failed to extract audio: {res.stderr}")
            raise RuntimeError(f"Failed to extract audio track: {res.stderr}")
            
        # 2. Whisper Transcription
        logger.info(f"Transcribing audio with Whisper (Source Language: {source_lang})...")
        source_srt_path = os.path.join(task_dir, "source_subtitles.srt")
        
        # Map human readable name to Whisper codes
        whisper_lang = None
        if source_lang.lower() in ["english", "en"]:
            whisper_lang = "en"
        elif source_lang.lower() in ["chinese", "zh"]:
            whisper_lang = "zh"
            
        create_subtitle(audio_path, source_srt_path, language=whisper_lang)
        
        if not os.path.exists(source_srt_path) or os.path.getsize(source_srt_path) == 0:
            logger.error("Whisper transcription did not generate any subtitles.")
            raise RuntimeError("Transcription failed: empty subtitle file generated.")
            
        # 3. Release Whisper model to free memory
        release_whisper_model()
        
        # 4. Translate SRT to Vietnamese using Gemini
        logger.info("Translating subtitles to Vietnamese via Gemini API...")
        translated_srt_path = os.path.join(task_dir, "vietnamese_subtitles.srt")
        translate_srt(
            srt_path=source_srt_path,
            output_path=translated_srt_path,
            source_lang=source_lang,
            target_lang="Vietnamese"
        )
        
        # 5. Optional Dubbing (Voiceover) Generation
        dubbed_audio_path = None
        if enable_voiceover:
            from app.services.dubbing import generate_dubbed_audio
            logger.info("Lồng tiếng (Voiceover) enabled. Starting TTS dubbing track generation...")
            dubbed_audio_path = generate_dubbed_audio(
                task_id=task_id,
                video_path=video_path,
                translated_srt_path=translated_srt_path,
                source_srt_path=source_srt_path,
                engine_name=tts_engine,
                voice_name=tts_voice,
                ducking_ratio=ducking_ratio,
                auto_clone=auto_clone
            )
            
        # 6. Burn subtitles into video
        final_video_path = os.path.join(task_dir, "translated_video.mp4")
        
        # Target audio path is dubbed audio if available, otherwise original extracted audio is handled implicitly or none
        # (For FFmpeg filter, if no dubbed_audio_path, we pass audio_path=None so it copies original audio directly).
        target_audio = dubbed_audio_path
        
        if burn_method == "ffmpeg":
            logger.info("Burning subtitles into video via FFmpeg filter...")
            success = burn_subtitles_ffmpeg(
                video_path=video_path,
                subtitle_path=translated_srt_path,
                output_path=final_video_path,
                audio_path=target_audio
            )
            if not success:
                logger.warning("FFmpeg native subtitle burning failed. Falling back to MoviePy...")
                burn_method = "moviepy"
                
        if burn_method == "moviepy":
            logger.info("Burning subtitles into video via MoviePy...")
            # We construct standard VideoParams to match generate_video
            from app.models.schema import VideoParams, VideoAspect
            
            # Retrieve video's original dimensions to determine aspect ratio parameter
            # MoviePy will adjust. Let's inspect video first
            from moviepy.video.io.VideoFileClip import VideoFileClip
            try:
                clip = VideoFileClip(video_path)
                w, h = clip.size
                clip.close()
                if w / h > 1.2:
                    aspect = VideoAspect.landscape
                elif h / w > 1.2:
                    aspect = VideoAspect.portrait
                else:
                    aspect = VideoAspect.square
            except Exception:
                aspect = VideoAspect.portrait
                
            bg_style = config.whisper.get("background_style", "None")
            bg_color = False if bg_style == "None" else config.whisper.get("text_background_color", "#000000")
            
            params = VideoParams(
                video_aspect=aspect,
                video_concat_mode=VideoConcatMode.random,
                voice_volume=1.0,
                bgm_type="",
                bgm_file="",
                bgm_volume=0.0,
                subtitle_enabled=True,
                font_name=config.whisper.get("font_name", "STHeitiMedium.ttc"),
                font_size=int(config.whisper.get("font_size", 45)), # Slightly smaller by default for subtitles
                text_fore_color=config.whisper.get("text_fore_color", "#FFFFFF"),
                stroke_color=config.whisper.get("stroke_color", "#000000"),
                stroke_width=float(config.whisper.get("stroke_width", 1.5)),
                text_background_color=bg_color,
                subtitle_bg_alpha=int(config.whisper.get("subtitle_bg_alpha", 140)),
                rounded_subtitle_background=bool(config.whisper.get("rounded_subtitle_background", False)),
                subtitle_position=config.whisper.get("subtitle_position", "bottom"),
                custom_position=float(config.whisper.get("custom_position", 70.0)),
                n_threads=os.cpu_count() or 4
            )
            
            # If voiceover is enabled, MoviePy should overlay subtitles and use dubbed audio track,
            # otherwise it overlays subtitles and uses original extracted audio path.
            moviepy_audio = target_audio if target_audio else audio_path
            
            generate_video(
                video_path=video_path,
                audio_path=moviepy_audio,
                subtitle_path=translated_srt_path,
                output_file=final_video_path,
                params=params
            )
            
        logger.success(f"Automatic translation workflow completed! Final video: {final_video_path}")
        return final_video_path

composer = ComposerWorkflow()
