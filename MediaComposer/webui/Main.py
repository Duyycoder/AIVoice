import os
import tempfile
import sys
# Redirect temp files to F drive to prevent C drive overloading/stutter
custom_temp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "temp")
os.makedirs(custom_temp, exist_ok=True)
os.environ["TEMP"] = custom_temp
os.environ["TMP"] = custom_temp
tempfile.tempdir = custom_temp
import uuid
import random
import shutil
import urllib.request
import streamlit as st
from urllib.parse import urlparse
from loguru import logger
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.config import config
from app.models.schema import VideoAspect, VideoConcatMode
from app.services.composer import composer
from app.utils import utils

st.set_page_config(page_title="MediaComposer", page_icon="🎬", layout="wide")

st.title("🎬 MediaComposer Standalone")
st.write("Generate a fully voiceovered video from audio files.")

# Basic Setup
with st.sidebar:
    st.header("Global Settings")
    
    # LLM Provider selection
    providers = ["OpenAI", "Google Gemini"]
    current_provider = config.app.get("llm_provider", "OpenAI")
    if current_provider not in providers:
        current_provider = "OpenAI"
    llm_provider = st.selectbox("LLM Provider", providers, index=providers.index(current_provider))
    
    # Configure defaults based on provider
    if llm_provider == "OpenAI":
        api_key_label = "OpenAI API Key"
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
    else:
        api_key_label = "Gemini API Key"
        default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        default_model = "gemini-2.0-flash"
        
    api_key_val = config.app.get("openai_api_key", "")
    api_key = st.text_input(api_key_label, value=api_key_val, type="password")
    
    # Configure model name
    model_val = config.app.get("openai_model", default_model)
    if llm_provider != config.app.get("llm_provider"):
        model_val = default_model
    if llm_provider == "Google Gemini" and model_val == "gemini-1.5-flash":
        model_val = "gemini-2.0-flash"
        
    llm_model = st.text_input("LLM Model Name", value=model_val)
    
    # Save if changed
    if (llm_provider != config.app.get("llm_provider") or 
        api_key != config.app.get("openai_api_key") or
        llm_model != config.app.get("openai_model")):
        config.app["llm_provider"] = llm_provider
        config.app["openai_api_key"] = api_key
        config.app["openai_base_url"] = default_base_url
        config.app["openai_model"] = llm_model
        config.save_config()

    st.header("Output Settings")
    if "output_folder" not in st.session_state:
        st.session_state["output_folder"] = root_dir

    col_dir, col_btn = st.columns([4, 1])
    with col_btn:
        st.write("")  # Spacer to align with text input label
        st.write("")
        if st.button("📁", help="Chọn thư mục đầu ra"):
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            selected = filedialog.askdirectory(parent=root)
            root.destroy()
            if selected:
                st.session_state["output_folder"] = os.path.normpath(selected)

    with col_dir:
        output_folder = st.text_input("Output Folder (Thư mục lưu)", value=st.session_state["output_folder"])
        if output_folder != st.session_state["output_folder"]:
            st.session_state["output_folder"] = output_folder

    output_filename = st.text_input("Output File Name (Tên file video)", value="output.mp4")
    output_path = os.path.join(st.session_state["output_folder"], output_filename)

    st.header("BGM Settings")
    enable_bgm = st.checkbox("Enable BGM", value=False)
    bgm_file = ""
    if enable_bgm:
        bgm_mode = st.radio("BGM Option", ["Default BGM (Các file mặc định)", "Auto-find New BGM (Tự tìm cái mới)", "Upload BGM (Tải lên)"])
        if bgm_mode == "Default BGM (Các file mặc định)":
            songs = []
            song_dir = utils.song_dir()
            if os.path.exists(song_dir):
                for file in os.listdir(song_dir):
                    if file.endswith(".mp3"):
                        songs.append(file)
            if songs:
                bgm_selection = st.selectbox("Select BGM", songs)
                bgm_file = bgm_selection
            else:
                st.warning("No BGM files found in resource/songs.")
        elif bgm_mode == "Auto-find New BGM (Tự tìm cái mới)":
            download_btn = st.button("Tải nhạc mới từ Internet")
            if download_btn:
                with st.spinner("Downloading background music..."):
                    try:
                        song_num = random.randint(1, 16)
                        url = f"https://www.soundhelix.com/examples/mp3/SoundHelix-Song-{song_num}.mp3"
                        dest_dir = utils.song_dir()
                        os.makedirs(dest_dir, exist_ok=True)
                        filename = f"soundhelix_song_{song_num}.mp3"
                        dest_path = os.path.join(dest_dir, filename)
                        if not os.path.exists(dest_path):
                            req = urllib.request.Request(
                                url, 
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                            )
                            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                                out_file.write(response.read())
                        st.session_state["auto_bgm_file"] = filename
                        st.success(f"Downloaded and selected: {filename}")
                    except Exception as e:
                        st.error(f"Failed to download BGM: {e}")
            auto_bgm = st.session_state.get("auto_bgm_file", "")
            if auto_bgm:
                st.info(f"Selected: {auto_bgm}")
                bgm_file = auto_bgm
            else:
                st.info("Click the button above to download and select new music.")
        elif bgm_mode == "Upload BGM (Tải lên)":
            bgm_upload = st.file_uploader("Upload BGM Audio", type=["mp3", "wav"])
            if bgm_upload:
                dest_dir = utils.song_dir()
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, bgm_upload.name)
                with open(dest_path, "wb") as f:
                    f.write(bgm_upload.getbuffer())
                bgm_file = bgm_upload.name
                st.success(f"Uploaded and selected: {bgm_upload.name}")

    st.header("Whisper Settings")
    
    # 1. Device selection
    whisper_device_opts = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    current_device = config.whisper.get("device", "cpu")
    if current_device not in whisper_device_opts:
        current_device = whisper_device_opts[0]
        
    whisper_device = st.selectbox("Whisper Device", whisper_device_opts, index=whisper_device_opts.index(current_device))
    
    # 2. Model Size selection
    whisper_model_opts = ["tiny", "base", "small", "medium", "large-v3"]
    current_model = config.whisper.get("model_size", "base")
    if current_model not in whisper_model_opts:
        current_model = "base"
        
    whisper_model = st.selectbox("Whisper Model", whisper_model_opts, index=whisper_model_opts.index(current_model))
    
    # 3. Compute Type selection
    if whisper_device == "cuda":
        compute_opts = ["float16", "int8_float16", "int8", "float32"]
    else:
        compute_opts = ["int8", "float32"]
        
    current_compute = config.whisper.get("compute_type", "int8")
    if current_compute not in compute_opts:
        current_compute = compute_opts[0]
        
    whisper_compute = st.selectbox("Compute Type", compute_opts, index=compute_opts.index(current_compute))
    
    # Save to config if changed
    if (whisper_device != config.whisper.get("device") or 
        whisper_model != config.whisper.get("model_size") or 
        whisper_compute != config.whisper.get("compute_type")):
        config.whisper["device"] = whisper_device
        config.whisper["model_size"] = whisper_model
        config.whisper["compute_type"] = whisper_compute
        config.save_config()
        # Reset cached whisper model to reload with new settings
        try:
            import app.services.subtitle as subtitle
            subtitle.model = None
        except Exception:
            pass

    st.header("Video Settings")
    codecs = ["libx264", "h264_nvenc"] if torch.cuda.is_available() else ["libx264"]
    current_codec = config.app.get("video_codec", "libx264")
    if current_codec not in codecs:
        current_codec = "libx264"
    video_codec = st.selectbox("Video Encoder", codecs, index=codecs.index(current_codec), help="Chọn h264_nvenc để xuất video bằng GPU NVIDIA (nhanh hơn & giảm tải CPU)")
    if video_codec != config.app.get("video_codec"):
        config.app["video_codec"] = video_codec
        config.save_config()

tab1, tab2 = st.tabs(["Manual Mode (Upload)", "Auto Mode (Fetch)"])

def save_uploaded_file(uploaded_file, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    file_path = os.path.join(dest_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def merge_audio_files(uploaded_audios, task_dir):
    if not uploaded_audios:
        return None
        
    # Sort alphabetically from A to Z by file name
    sorted_audios = sorted(uploaded_audios, key=lambda x: x.name)
    
    # Save files to task directory
    saved_paths = []
    for f in sorted_audios:
        saved_paths.append(save_uploaded_file(f, task_dir))
        
    if len(saved_paths) == 1:
        return saved_paths[0]
        
    # Concat multiple audio files
    logger.info("Merging multiple audio files in A-Z order...")
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.audio.AudioClip import concatenate_audioclips
    
    merged_path = os.path.join(task_dir, "merged_input_audio.mp3")
    clips = []
    try:
        for p in saved_paths:
            clips.append(AudioFileClip(p))
        final_clip = concatenate_audioclips(clips)
        final_clip.write_audiofile(merged_path, logger=None)
        final_clip.close()
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
    return merged_path

with tab1:
    st.header("Workflow 1: Manual Mode")
    st.write("Upload your audio(s) and video/images. Visuals will loop randomly to match audio length. Images show for 4s.")
    
    col1, col2 = st.columns(2)
    with col1:
        audio_files_m = st.file_uploader("Upload Audio(s) (Required)", type=["mp3", "wav", "m4a"], accept_multiple_files=True, key="m_audio")
    with col2:
        video_files_m = st.file_uploader("Upload Videos/Images", type=["mp4", "mov", "jpg", "png"], accept_multiple_files=True, key="m_videos")
    
    aspect_m = st.selectbox("Aspect Ratio", [VideoAspect.portrait.value, VideoAspect.landscape.value, VideoAspect.square.value], key="m_aspect")
    subtitles_m = st.checkbox("Enable Whisper Subtitles", value=True, key="m_subs")
    
    if st.button("Generate Video (Manual)", type="primary"):
        if not audio_files_m:
            st.error("Audio file is required!")
        elif not video_files_m:
            st.error("At least one video/image is required!")
        else:
            task_id = str(uuid.uuid4())
            task_dir = utils.task_dir(task_id)
            os.makedirs(task_dir, exist_ok=True)
            log_path = os.path.join(task_dir, "run.log")
            
            # Start loguru sink
            sink_id = logger.add(log_path, format="{time:HH:mm:ss} - {message}", level="INFO")
            
            st.info("Bắt đầu xử lý... Xem tiến độ ở khung Console Log bên dưới.")
            log_placeholder = st.empty()
            
            import threading
            import time
            
            result = {"video": None, "error": None, "done": False}
            
            def run_in_thread():
                try:
                    logger.info("=== Bắt đầu Workflow 1 (Manual Mode) ===")
                    audio_path = merge_audio_files(audio_files_m, task_dir)
                    video_paths = []
                    for v in video_files_m:
                        video_paths.append(save_uploaded_file(v, task_dir))
                        
                    final_video = composer.run_workflow(
                        task_id=task_id,
                        audio_path=audio_path,
                        video_paths=video_paths,
                        auto_fetch=False,
                        bgm_file=bgm_file if enable_bgm else "",
                        video_aspect=VideoAspect(aspect_m),
                        concat_mode=VideoConcatMode.random,
                        enable_subtitles=subtitles_m
                    )
                    result["video"] = final_video
                    logger.info("=== Hoàn thành Workflow 1 thành công! ===")
                except Exception as ex:
                    import traceback
                    logger.error(f"Lỗi: {ex}")
                    logger.error(traceback.format_exc())
                    result["error"] = ex
                finally:
                    try:
                        from app.services.subtitle import release_whisper_model
                        release_whisper_model()
                    except Exception as clean_ex:
                        logger.warning(f"Lỗi giải phóng VRAM: {clean_ex}")
                    result["done"] = True
                    try:
                        logger.remove(sink_id)
                    except Exception:
                        pass
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            
            # Read and display log in real-time
            while not result["done"]:
                time.sleep(0.5)
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        log_data = f.read()
                    log_placeholder.code(log_data, language="text")
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                log_placeholder.code(log_data, language="text")
                
            if result["error"]:
                st.error(f"Error during video generation: {result['error']}")
            elif result["video"]:
                final_video = result["video"]
                if output_path:
                    out_dir = os.path.dirname(os.path.abspath(output_path))
                    os.makedirs(out_dir, exist_ok=True)
                    shutil.copy(final_video, output_path)
                    st.success(f"Video generated and saved to: {output_path}")
                    st.video(output_path)
                else:
                    st.success("Video generated successfully!")
                    st.video(final_video)

with tab2:
    st.header("Workflow 2: Auto Fetch")
    st.write("Upload audio(s). The system will transcribe it, use LLM to infer keywords, and fetch videos automatically.")
    
    audio_files_a = st.file_uploader("Upload Audio(s) (Required)", type=["mp3", "wav", "m4a"], accept_multiple_files=True, key="a_audio")
    source_a = st.selectbox("Source", ["pexels", "pixabay", "coverr"], key="a_source")
    
    # Dynamic Source API Key Input field
    if source_a == "pexels":
        api_keys = config.app.get("pexels_api_keys", "")
        val = api_keys[0] if (isinstance(api_keys, list) and api_keys) else (api_keys if isinstance(api_keys, str) else "")
        pexels_key = st.text_input("Pexels API Key", value=val, type="password", key="pexels_key_input")
        if pexels_key != val:
            config.app["pexels_api_keys"] = pexels_key
            config.save_config()
            
    elif source_a == "pixabay":
        api_keys = config.app.get("pixabay_api_keys", "")
        val = api_keys[0] if (isinstance(api_keys, list) and api_keys) else (api_keys if isinstance(api_keys, str) else "")
        pixabay_key = st.text_input("Pixabay API Key", value=val, type="password", key="pixabay_key_input")
        if pixabay_key != val:
            config.app["pixabay_api_keys"] = pixabay_key
            config.save_config()
            
    elif source_a == "coverr":
        api_keys = config.app.get("coverr_api_keys", "")
        val = api_keys[0] if (isinstance(api_keys, list) and api_keys) else (api_keys if isinstance(api_keys, str) else "")
        coverr_key = st.text_input("Coverr API Key", value=val, type="password", key="coverr_key_input")
        if coverr_key != val:
            config.app["coverr_api_keys"] = coverr_key
            config.save_config()
            
    aspect_a = st.selectbox("Aspect Ratio", [VideoAspect.portrait.value, VideoAspect.landscape.value, VideoAspect.square.value], key="a_aspect")
    subtitles_a = st.checkbox("Enable Whisper Subtitles", value=True, key="a_subs")
    
    if st.button("Generate Video (Auto)", type="primary"):
        if not audio_files_a:
            st.error("Audio file is required!")
        else:
            task_id = str(uuid.uuid4())
            task_dir = utils.task_dir(task_id)
            os.makedirs(task_dir, exist_ok=True)
            log_path = os.path.join(task_dir, "run.log")
            
            # Start loguru sink
            sink_id = logger.add(log_path, format="{time:HH:mm:ss} - {message}", level="INFO")
            
            st.info("Bắt đầu xử lý... Xem tiến độ ở khung Console Log bên dưới.")
            log_placeholder = st.empty()
            
            import threading
            import time
            
            result = {"video": None, "error": None, "done": False}
            
            def run_in_thread():
                try:
                    logger.info("=== Bắt đầu Workflow 2 (Auto Mode) ===")
                    audio_path = merge_audio_files(audio_files_a, task_dir)
                    
                    final_video = composer.run_workflow(
                        task_id=task_id,
                        audio_path=audio_path,
                        auto_fetch=True,
                        source=source_a,
                        bgm_file=bgm_file if enable_bgm else "",
                        video_aspect=VideoAspect(aspect_a),
                        concat_mode=VideoConcatMode.random,
                        enable_subtitles=subtitles_a
                    )
                    result["video"] = final_video
                    logger.info("=== Hoàn thành Workflow 2 thành công! ===")
                except Exception as ex:
                    import traceback
                    logger.error(f"Lỗi: {ex}")
                    logger.error(traceback.format_exc())
                    result["error"] = ex
                finally:
                    try:
                        from app.services.subtitle import release_whisper_model
                        release_whisper_model()
                    except Exception as clean_ex:
                        logger.warning(f"Lỗi giải phóng VRAM: {clean_ex}")
                    result["done"] = True
                    try:
                        logger.remove(sink_id)
                    except Exception:
                        pass
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            
            # Read and display log in real-time
            while not result["done"]:
                time.sleep(0.5)
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        log_data = f.read()
                    log_placeholder.code(log_data, language="text")
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                log_placeholder.code(log_data, language="text")
                
            if result["error"]:
                st.error(f"Error during video generation: {result['error']}")
            elif result["video"]:
                final_video = result["video"]
                if output_path:
                    out_dir = os.path.dirname(os.path.abspath(output_path))
                    os.makedirs(out_dir, exist_ok=True)
                    shutil.copy(final_video, output_path)
                    st.success(f"Video generated and saved to: {output_path}")
                    st.video(output_path)
                else:
                    st.success("Video generated successfully!")
                    st.video(final_video)
