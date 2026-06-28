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
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.config import config
from app.models.schema import VideoAspect, VideoConcatMode
from app.services.composer import composer
from app.utils import utils

st.set_page_config(page_title="MediaComposer", page_icon="🎬", layout="wide")

st.title("🎬 MediaComposer Standalone")
st.write("Generate a fully voiceovered video from audio files.")

# --- Task Manager System ---
import datetime
import time
import threading

def get_all_tasks():
    tasks_dir = utils.storage_dir("tasks")
    if not os.path.exists(tasks_dir):
        return []
    tasks_list = []
    try:
        subdirs = os.listdir(tasks_dir)
    except Exception:
        return []
        
    for task_id in subdirs:
        try:
            task_dir = os.path.join(tasks_dir, task_id)
            if not os.path.isdir(task_dir):
                continue
            log_path = os.path.join(task_dir, "run.log")
            if not os.path.exists(log_path):
                continue
                
            thread_name = f"task_{task_id}"
            is_running = any(t.name == thread_name for t in threading.enumerate())
            mtime = os.path.getmtime(log_path)
            
            final_video = os.path.join(task_dir, "final.mp4")
            split_manifest = os.path.join(task_dir, "split_manifest.json")
            has_video = os.path.exists(final_video) or os.path.exists(split_manifest)
            
            tasks_list.append({
                "id": task_id,
                "path": task_dir,
                "log_path": log_path,
                "is_running": is_running,
                "has_video": has_video,
                "mtime": mtime
            })
        except Exception:
            continue
            
    try:
        tasks_list.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        pass
    return tasks_list

tasks = get_all_tasks()
running_tasks = [t for t in tasks if t["is_running"]]

# Render selected task detail view if active
if "selected_task_id" in st.session_state:
    selected_task_id = st.session_state["selected_task_id"]
    task_info = next((t for t in tasks if t["id"] == selected_task_id), None)
    if task_info:
        st.markdown(f"### 📊 Chi tiết tác vụ `{selected_task_id}`")
        
        col_back, col_del = st.columns([1, 1])
        with col_back:
            if st.button("⬅️ Quay lại danh sách", use_container_width=True):
                del st.session_state["selected_task_id"]
                st.rerun()
        with col_del:
            if st.button("🗑️ Xóa sạch tác vụ này", type="primary", use_container_width=True):
                shutil.rmtree(task_info["path"], ignore_errors=True)
                del st.session_state["selected_task_id"]
                st.success("Đã xóa dữ liệu tác vụ!")
                st.rerun()
                
        log_placeholder = st.empty()
        
        is_running = any(t.name == f"task_{selected_task_id}" for t in threading.enumerate())
        if is_running:
            st.info("⚡ Tác vụ đang xử lý ngầm... Log sẽ tự động cuộn xuống dưới.")
            while any(t.name == f"task_{selected_task_id}" for t in threading.enumerate()):
                time.sleep(1.0)
                if os.path.exists(task_info["log_path"]):
                    with open(task_info["log_path"], "r", encoding="utf-8", errors="ignore") as f:
                        log_data = f.read()
                    try:
                        log_placeholder.code(log_data, language="text")
                    except Exception:
                        break
            st.rerun()
        else:
            if os.path.exists(task_info["log_path"]):
                with open(task_info["log_path"], "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                log_placeholder.code(log_data, language="text")
                
            split_manifest_path = os.path.join(task_info["path"], "split_manifest.json")
            if os.path.exists(split_manifest_path):
                st.success("🎉 Tác vụ chia video hoàn thành thành công!")
                import json
                try:
                    with open(split_manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    parts = manifest.get("parts", [])
                    st.write(f"Đã chia video thành {len(parts)} phần:")
                    zip_path = os.path.join(task_info["path"], "split_parts.zip")
                    if os.path.exists(zip_path):
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="📥 Tải xuống tất cả các phần (ZIP)",
                                data=f,
                                file_name="split_parts.zip",
                                mime="application/zip",
                                key=f"dl_zip_{selected_task_id}"
                            )
                    for idx, part_filename in enumerate(parts):
                        part_abs_path = os.path.join(task_info["path"], part_filename)
                        if os.path.exists(part_abs_path):
                            st.markdown(f"**Phần {idx+1}: {part_filename}**")
                            st.video(part_abs_path)
                            with open(part_abs_path, "rb") as f:
                                st.download_button(
                                    label=f"📥 Tải xuống phần {idx+1}",
                                    data=f,
                                    file_name=part_filename,
                                    mime="video/mp4",
                                    key=f"dl_part_{idx}_{selected_task_id}"
                                )
                except Exception as e:
                    st.error(f"Lỗi đọc kết quả chia video: {e}")
            elif task_info["has_video"]:
                st.success("🎉 Tác vụ hoàn thành thành công!")
                st.video(os.path.join(task_info["path"], "final.mp4"))
            else:
                st.error("❌ Tác vụ đã dừng hoặc gặp lỗi (Xem log ở trên để biết chi tiết).")
        st.divider()

# Expandable Task Manager
if running_tasks:
    st.warning(f"⚠️ Phát hiện {len(running_tasks)} tác vụ đang chạy ngầm trên server!")

with st.expander("📊 Quản lý tác vụ chạy ngầm (Task Manager)", expanded=bool(running_tasks)):
    if not tasks:
        st.write("Chưa có tác vụ nào được ghi nhận.")
    else:
        for t in tasks:
            status_str = "🟢 Đang chạy" if t["is_running"] else ("🔵 Đã xong" if t["has_video"] else "🔴 Đã dừng/Lỗi")
            time_str = datetime.datetime.fromtimestamp(t["mtime"]).strftime('%Y-%m-%d %H:%M:%S')
            
            col_id, col_status, col_time, col_action = st.columns([3, 1, 2, 2])
            with col_id:
                st.write(f"`{t['id']}`")
            with col_status:
                st.write(status_str)
            with col_time:
                st.write(time_str)
            with col_action:
                if st.button("Xem chi tiết & Log", key=f"btn_{t['id']}", use_container_width=True):
                    st.session_state["selected_task_id"] = t["id"]
                    st.rerun()
st.divider()


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
            import subprocess
            import sys
            cmd = [
                sys.executable,
                "-c",
                "import sys; sys.stdout.reconfigure(encoding='utf-8'); import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); root.attributes('-topmost', True); print(filedialog.askdirectory(title='Chọn thư mục')); root.destroy()"
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                selected = res.stdout.strip()
            except Exception as e:
                logger.error(f"Error in dir dialog subprocess: {e}")
                selected = ""
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

    st.header("Subtitle Style Settings")
    
    # Quét font động từ thư mục resource/fonts
    import os
    font_dir = utils.font_dir()
    os.makedirs(font_dir, exist_ok=True)
    try:
        available_fonts = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.ttc'))]
    except Exception:
        available_fonts = []
    if not available_fonts:
        available_fonts = ["STHeitiMedium.ttc", "Arial-Regular.ttf"]
        
    current_font = config.whisper.get("font_name", "STHeitiMedium.ttc")
    if current_font not in available_fonts:
        current_font = available_fonts[0]
    sub_font = st.selectbox("Font chữ (Font)", available_fonts, index=available_fonts.index(current_font))
    
    current_font_size = int(config.whisper.get("font_size", 60))
    sub_font_size = st.slider("Kích thước (Size)", min_value=20, max_value=120, value=current_font_size)
    
    current_fore_color = config.whisper.get("text_fore_color", "#FFFFFF")
    sub_fore_color = st.color_picker("Màu chữ (Text Color)", value=current_fore_color)
    
    current_stroke_color = config.whisper.get("stroke_color", "#000000")
    sub_stroke_color = st.color_picker("Màu viền (Stroke Color)", value=current_stroke_color)
    
    current_stroke_width = float(config.whisper.get("stroke_width", 1.5))
    sub_stroke_width = st.slider("Độ dày viền (Stroke Width)", min_value=0.0, max_value=5.0, value=current_stroke_width, step=0.1)
    
    current_bg_style = config.whisper.get("background_style", "None")
    bg_style_opts = ["None", "Black", "Custom"]
    bg_style_index = bg_style_opts.index(current_bg_style) if current_bg_style in bg_style_opts else 0
    sub_bg_style = st.selectbox("Kiểu nền (Background)", ["Không nền", "Nền đen mặc định", "Nền màu tùy chọn"], index=bg_style_index)
    bg_style_val = bg_style_opts[["Không nền", "Nền đen mặc định", "Nền màu tùy chọn"].index(sub_bg_style)]
    
    sub_bg_color = "#000000"
    if bg_style_val == "Custom":
        current_bg_color = config.whisper.get("text_background_color", "#000000")
        sub_bg_color = st.color_picker("Màu nền (Background Color)", value=current_bg_color)
        
    has_bg = (bg_style_val != "None")
    
    sub_bg_alpha = 140
    sub_rounded = False
    if has_bg:
        default_alpha = 140 if bg_style_val == "Black" else 255
        current_bg_alpha = int(config.whisper.get("subtitle_bg_alpha", default_alpha))
        sub_bg_alpha_pct = st.slider("Độ mờ nền (%)", min_value=0, max_value=100, value=int(current_bg_alpha / 255.0 * 100))
        sub_bg_alpha = int(sub_bg_alpha_pct / 100.0 * 255)
        
        current_rounded = bool(config.whisper.get("rounded_subtitle_background", False))
        sub_rounded = st.checkbox("Bo góc nền (Rounded corners)", value=current_rounded)
        
    current_pos_style = config.whisper.get("subtitle_position", "bottom")
    pos_opts = ["bottom", "custom"]
    pos_style_index = pos_opts.index(current_pos_style) if current_pos_style in pos_opts else 0
    sub_pos_style = st.selectbox("Vị trí (Position)", ["Mặc định (Dưới)", "Tùy chọn (Custom)"], index=pos_style_index)
    pos_style_val = pos_opts[["Mặc định (Dưới)", "Tùy chọn (Custom)"].index(sub_pos_style)]
    
    sub_custom_pos = 70.0
    if pos_style_val == "custom":
        current_custom_pos = float(config.whisper.get("custom_position", 70.0))
        sub_custom_pos = st.slider("Chiều cao vị trí (%)", min_value=10.0, max_value=90.0, value=current_custom_pos, step=1.0)
        
    # Lưu cấu hình phụ đề
    if (sub_font != config.whisper.get("font_name") or
        sub_font_size != config.whisper.get("font_size") or
        sub_fore_color != config.whisper.get("text_fore_color") or
        sub_stroke_color != config.whisper.get("stroke_color") or
        sub_stroke_width != config.whisper.get("stroke_width") or
        bg_style_val != config.whisper.get("background_style") or
        (bg_style_val == "Custom" and sub_bg_color != config.whisper.get("text_background_color")) or
        (has_bg and (sub_bg_alpha != config.whisper.get("subtitle_bg_alpha") or sub_rounded != config.whisper.get("rounded_subtitle_background"))) or
        pos_style_val != config.whisper.get("subtitle_position") or
        (pos_style_val == "custom" and sub_custom_pos != config.whisper.get("custom_position"))):
        
        config.whisper["font_name"] = sub_font
        config.whisper["font_size"] = sub_font_size
        config.whisper["text_fore_color"] = sub_fore_color
        config.whisper["stroke_color"] = sub_stroke_color
        config.whisper["stroke_width"] = sub_stroke_width
        config.whisper["background_style"] = bg_style_val
        if bg_style_val == "Custom":
            config.whisper["text_background_color"] = sub_bg_color
        if has_bg:
            config.whisper["subtitle_bg_alpha"] = sub_bg_alpha
            config.whisper["rounded_subtitle_background"] = sub_rounded
        config.whisper["subtitle_position"] = pos_style_val
        if pos_style_val == "custom":
            config.whisper["custom_position"] = sub_custom_pos
            
        config.save_config()

    # Bộ xem trước trực tiếp (Live Preview)
    st.markdown("### Live Preview (Xem trước)")
    css_text_color = sub_fore_color
    
    if bg_style_val == "None":
        css_bg = "background-color: transparent;"
    elif bg_style_val == "Black":
        alpha_float = round(sub_bg_alpha / 255.0, 2)
        css_bg = f"background-color: rgba(0, 0, 0, {alpha_float});"
    else:
        hex_color = sub_bg_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        alpha_float = round(sub_bg_alpha / 255.0, 2)
        css_bg = f"background-color: rgba({r}, {g}, {b}, {alpha_float});"
        
    css_radius = "border-radius: 8px;" if sub_rounded else "border-radius: 0px;"
    css_padding = "padding: 6px 12px;" if has_bg else "padding: 0px;"
    
    st.markdown(
        f"""
        <div style="
            background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('https://images.pexels.com/photos/15286/pexels-photo.jpg?auto=compress&cs=tinysrgb&h=150');
            background-size: cover;
            background-position: center;
            height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            border: 1px solid #555;
            position: relative;
        ">
            <span style="
                color: {css_text_color};
                font-size: {int(sub_font_size * 0.35)}px;
                font-weight: bold;
                text-align: center;
                text-shadow: 
                    -{sub_stroke_width}px -{sub_stroke_width}px 0 {sub_stroke_color},  
                     {sub_stroke_width}px -{sub_stroke_width}px 0 {sub_stroke_color},
                    -{sub_stroke_width}px  {sub_stroke_width}px 0 {sub_stroke_color},
                     {sub_stroke_width}px  {sub_stroke_width}px 0 {sub_stroke_color};
                {css_bg}
                {css_radius}
                {css_padding}
                max-width: 90%;
                word-wrap: break-word;
                line-height: 1.2;
            ">
                Đây là phụ đề mẫu
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

tab1, tab2, tab3 = st.tabs(["Manual Mode (Upload)", "Auto Mode (Fetch)", "Split Video (Workflow 3)"])

def save_uploaded_file(uploaded_file, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    file_path = os.path.join(dest_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def merge_audio_files(uploaded_audios, task_dir):
    """
    Ghép nhiều file audio thành một file duy nhất.
    Ưu tiên dùng FFmpeg concat demuxer (-c copy) — không decode/re-encode,
    nhanh hơn MoviePy ~50-100x với batch lớn.
    Fallback về MoviePy nếu FFmpeg không có trong PATH.
    """
    if not uploaded_audios:
        return None

    # Natural sort: tách cụm số và so sánh dưới dạng int,
    # tránh lỗi "Chương 111" xếp trước "Chương 89"
    import re
    def _natural_sort_key(item):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', item.name)]
    sorted_audios = sorted(uploaded_audios, key=_natural_sort_key)

    saved_paths = [save_uploaded_file(f, task_dir) for f in sorted_audios]
    if len(saved_paths) == 1:
        return saved_paths[0]

    sorted_names = [f.name for f in sorted_audios]
    logger.info(f"Merging {len(sorted_names)} audio files: {sorted_names}")

    return _merge_audio_ffmpeg(saved_paths, task_dir)


def merge_audio_paths(audio_paths, task_dir):
    """
    Ghép nhiều file audio từ đường dẫn cục bộ thành một file duy nhất.
    """
    if not audio_paths:
        return None

    import re
    def _natural_sort_key(path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', os.path.basename(path))]
    sorted_paths = sorted(audio_paths, key=_natural_sort_key)
    if len(sorted_paths) == 1:
        return sorted_paths[0]

    return _merge_audio_ffmpeg(sorted_paths, task_dir)


def _merge_audio_ffmpeg(saved_paths, task_dir):
    """
    Ghép bằng FFmpeg concat demuxer với -c copy (stream-copy).
    Vì tất cả file WAV đầu vào có cùng định dạng (cùng sample rate, channels, bit depth),
    stream-copy hoạt động trực tiếp mà không cần decode hay re-encode.
    Fallback về MoviePy chỉ khi FFmpeg không tồn tại hoặc gặp lỗi bất ngờ.
    """
    import subprocess
    import shutil as _shutil

    # Tìm ffmpeg: ưu tiên system ffmpeg, fallback về imageio_ffmpeg (cài sẵn qua moviepy)
    ffmpeg_exe = _shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    if ffmpeg_exe is None:
        logger.warning("ffmpeg not found in PATH, falling back to MoviePy (slow).")
        return _merge_audio_moviepy(saved_paths, task_dir)

    # Ghi filelist với đường dẫn tuyệt đối (required by -safe 0)
    filelist_path = os.path.join(task_dir, "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as flist:
        for p in saved_paths:
            abs_p = os.path.abspath(p).replace("\\", "/")
            flist.write(f"file '{abs_p}'\n")

    # Output WAV — giữ nguyên định dạng đầu vào, stream-copy không mất chất lượng
    merged_path = os.path.join(task_dir, "merged_input_audio.wav")

    logger.info(f"[FFmpeg] Stream-copy {len(saved_paths)} files → merged_input_audio.wav")
    result = subprocess.run(
        [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", filelist_path, "-c", "copy", merged_path],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        logger.info("[FFmpeg] Merge complete.")
        return merged_path

    # Chỉ fallback MoviePy nếu FFmpeg gặp lỗi bất ngờ
    logger.error(f"[FFmpeg] concat failed: {result.stderr[-400:]}")
    logger.warning("Falling back to MoviePy...")
    return _merge_audio_moviepy(saved_paths, task_dir)


def _merge_audio_moviepy(saved_paths, task_dir):
    """Fallback: ghép audio bằng MoviePy (chậm hơn nhưng luôn hoạt động)."""
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.audio.AudioClip import concatenate_audioclips

    logger.info(f"[MoviePy] Merging {len(saved_paths)} files (this may take a while)...")
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
    st.write("Upload your audio(s) and video/images, or enter their local paths on your machine. Visuals will loop to match audio length.")
    
    input_method_m = st.radio(
        "Phương thức chọn file (Manual)",
        ["Upload qua trình duyệt", "Nhập đường dẫn cục bộ (Local Paths)"],
        horizontal=True,
        key="m_input_method"
    )
    
    audio_files_m = []
    video_files_m = []
    local_audio_paths_m = []
    local_video_paths_m = []
    
    if input_method_m == "Upload qua trình duyệt":
        col1, col2 = st.columns(2)
        with col1:
            audio_files_m = st.file_uploader("Upload Audio(s) (Required)", type=["mp3", "wav", "m4a"], accept_multiple_files=True, key="m_audio")
        with col2:
            video_files_m = st.file_uploader("Upload Videos/Images", type=["mp4", "mov", "jpg", "png"], accept_multiple_files=True, key="m_videos")
    else:
        col1, col2 = st.columns(2)
        with col1:
            local_audio_input = st.text_area("Đường dẫn file Audio cục bộ (Mỗi dòng một file, bắt buộc)", key="m_local_audio", help="Ví dụ: G:/Coding/AIVoice/data/inputs/example.wav")
            if local_audio_input.strip():
                local_audio_paths_m = [p.strip().strip('"').strip("'") for p in local_audio_input.split('\n') if p.strip()]
        with col2:
            local_video_input = st.text_area("Đường dẫn file Video/Image cục bộ (Mỗi dòng một file, bắt buộc)", key="m_local_video", help="Ví dụ: D:/videos/background.mp4")
            if local_video_input.strip():
                local_video_paths_m = [p.strip().strip('"').strip("'") for p in local_video_input.split('\n') if p.strip()]
                
    aspect_m = st.selectbox("Aspect Ratio", [VideoAspect.portrait.value, VideoAspect.landscape.value, VideoAspect.square.value], key="m_aspect")
    subtitles_m = st.checkbox("Enable Whisper Subtitles", value=True, key="m_subs")
    slice_video_m = st.checkbox("Cắt nhỏ video (Slicing)", value=True, key="m_slice", help="Nếu bật, video dài sẽ bị chia thành các clip ngắn (tối đa 30s) rồi ghép lại. Nếu tắt, hệ thống sẽ giữ nguyên thời lượng clip gốc của bạn.")
    
    has_audio = bool(audio_files_m) if input_method_m == "Upload qua trình duyệt" else bool(local_audio_paths_m)
    has_video = bool(video_files_m) if input_method_m == "Upload qua trình duyệt" else bool(local_video_paths_m)
    
    if st.button("Generate Video (Manual)", type="primary"):
        if not has_audio:
            st.error("Audio file is required!")
        elif not has_video:
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
                    if input_method_m == "Upload qua trình duyệt":
                        audio_path = merge_audio_files(audio_files_m, task_dir)
                        video_paths = []
                        for v in video_files_m:
                            video_paths.append(save_uploaded_file(v, task_dir))
                    else:
                        audio_path = merge_audio_paths(local_audio_paths_m, task_dir)
                        import re
                        def _natural_sort_key(path):
                            return [int(part) if part.isdigit() else part.lower()
                                    for part in re.split(r'(\d+)', os.path.basename(path))]
                        video_paths = sorted(local_video_paths_m, key=_natural_sort_key)
                        
                    final_video = composer.run_workflow(
                        task_id=task_id,
                        audio_path=audio_path,
                        video_paths=video_paths,
                        auto_fetch=False,
                        bgm_file=bgm_file if enable_bgm else "",
                        video_aspect=VideoAspect(aspect_m),
                        concat_mode=VideoConcatMode.random,
                        enable_subtitles=subtitles_m,
                        slice_video=slice_video_m
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
            
            thread = threading.Thread(target=run_in_thread, name=f"task_{task_id}")
            thread.start()
            
            # Read and display log in real-time
            while not result["done"]:
                time.sleep(0.5)
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        log_data = f.read()
                    try:
                        log_placeholder.code(log_data, language="text")
                    except Exception:
                        break
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                try:
                    log_placeholder.code(log_data, language="text")
                except Exception:
                    pass
                
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
    st.write("Upload audio(s) or enter local paths. The system will transcribe it, use LLM to infer keywords, and fetch videos automatically.")
    st.caption("💡 Nếu bạn đã có file text/markdown gốc, hãy upload để bỏ qua Whisper → tiết kiệm VRAM và thời gian.")
    
    input_method_a = st.radio(
        "Phương thức chọn file (Auto)",
        ["Upload qua trình duyệt", "Nhập đường dẫn cục bộ (Local Paths)"],
        horizontal=True,
        key="a_input_method"
    )
    
    audio_files_a = []
    local_audio_paths_a = []
    
    if input_method_a == "Upload qua trình duyệt":
        audio_files_a = st.file_uploader("Upload Audio(s) (Required)", type=["mp3", "wav", "m4a"], accept_multiple_files=True, key="a_audio")
    else:
        local_audio_input_a = st.text_area("Đường dẫn file Audio cục bộ (Mỗi dòng một file, bắt buộc)", key="a_local_audio", help="Ví dụ: G:/Coding/AIVoice/data/inputs/example.wav")
        if local_audio_input_a.strip():
            local_audio_paths_a = [p.strip().strip('"').strip("'") for p in local_audio_input_a.split('\n') if p.strip()]

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
    subtitles_a = st.checkbox("Enable Subtitles", value=True, key="a_subs")
    
    # Transcript text file uploader — bypasses Whisper entirely
    transcript_files_a = st.file_uploader(
        "📝 Transcript Text File (Tùy chọn — Bỏ qua Whisper)",
        type=["md", "txt"],
        accept_multiple_files=True,
        key="a_transcript",
        help="Upload file .md hoặc .txt chứa nội dung gốc của audio. Bạn có thể chọn nhiều file, hệ thống sẽ tự động sắp xếp theo thứ tự natural sort."
    )
    
    # Read and clean transcript text
    transcript_text_a = ""
    if transcript_files_a:
        import re
        def _natural_sort_key_file(f):
            return [int(part) if part.isdigit() else part.lower()
                    for part in re.split(r'(\d+)', f.name)]
        sorted_transcript_files = sorted(transcript_files_a, key=_natural_sort_key_file)
        
        cleaned_contents = []
        for transcript_file in sorted_transcript_files:
            raw_text = transcript_file.read().decode("utf-8", errors="ignore")
            # Strip markdown formatting if .md file
            import re
            def _clean_markdown_simple(text: str) -> str:
                """Strip markdown formatting, keeping only readable text."""
                if not text:
                    return ""
                text = re.sub(r'```[\s\S]*?```', '', text)  # code blocks
                text = re.sub(r'`[^`]+`', '', text)  # inline code
                text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)  # images
                text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # links
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold
                text = re.sub(r'\*(.+?)\*', r'\1', text)  # italic
                text = re.sub(r'__(.+?)__', r'\1', text)  # bold alt
                text = re.sub(r'_(.+?)_', r'\1', text)  # italic alt
                text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)  # hr
                text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)  # blockquote
                text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # headers
                text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # bullets
                text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered
                # Collapse multiple blank lines
                text = re.sub(r'\n{3,}', '\n\n', text)
                return text.strip()
            
            cleaned = _clean_markdown_simple(raw_text)
            if cleaned:
                cleaned_contents.append(cleaned)
        
        if cleaned_contents:
            transcript_text_a = "\n\n".join(cleaned_contents)
            word_count = len(transcript_text_a.split())
            file_names_sorted = [f.name for f in sorted_transcript_files]
            st.success(f"⚡ Chế độ nhanh: Sẽ bỏ qua Whisper transcription ({word_count:,} từ đã đọc từ {len(sorted_transcript_files)} file: {file_names_sorted})")
        else:
            st.warning("Các file transcript đều rỗng hoặc không đọc được, sẽ dùng Whisper như bình thường.")
            
    has_audio_a = bool(audio_files_a) if input_method_a == "Upload qua trình duyệt" else bool(local_audio_paths_a)
    
    if st.button("Generate Video (Auto)", type="primary"):
        if not has_audio_a:
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
                    if input_method_a == "Upload qua trình duyệt":
                        audio_path = merge_audio_files(audio_files_a, task_dir)
                    else:
                        audio_path = merge_audio_paths(local_audio_paths_a, task_dir)
                    
                    final_video = composer.run_workflow(
                        task_id=task_id,
                        audio_path=audio_path,
                        auto_fetch=True,
                        source=source_a,
                        bgm_file=bgm_file if enable_bgm else "",
                        video_aspect=VideoAspect(aspect_a),
                        concat_mode=VideoConcatMode.random,
                        enable_subtitles=subtitles_a,
                        transcript_text=transcript_text_a
                    )
                    result["video"] = final_video
                    logger.info("=== Hoàn thành Workflow 2 thành công! ===")
                except Exception as ex:
                    import traceback
                    logger.error(f"Lỗi: {ex}")
                    logger.error(traceback.format_exc())
                    result["error"] = ex
                finally:
                    # Only release Whisper if it was actually used
                    if not transcript_text_a:
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
            
            thread = threading.Thread(target=run_in_thread, name=f"task_{task_id}")
            thread.start()
            
            # Read and display log in real-time
            while not result["done"]:
                time.sleep(0.5)
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        log_data = f.read()
                    try:
                        log_placeholder.code(log_data, language="text")
                    except Exception:
                        break
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                try:
                    log_placeholder.code(log_data, language="text")
                except Exception:
                    pass
                
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


with tab3:
    st.header("Workflow 3: Split Video")
    st.write("Chia nhỏ video thành số đoạn bằng nhau. Người dùng có thể upload video hoặc chọn đường dẫn video cục bộ.")
    
    input_method_s = st.radio(
        "Phương thức chọn file (Split)",
        ["Upload qua trình duyệt", "Nhập đường dẫn cục bộ (Local Paths)"],
        horizontal=True,
        key="s_input_method"
    )
    
    uploaded_video_s = None
    local_video_path_s = ""
    
    if input_method_s == "Upload qua trình duyệt":
        uploaded_video_s = st.file_uploader("Upload Video (Bắt buộc)", type=["mp4", "mov", "avi", "mkv"], key="s_uploaded_video")
    else:
        local_video_path_s = st.text_input("Đường dẫn file Video cục bộ (Bắt buộc)", key="s_local_video", help="Ví dụ: D:/videos/background.mp4")
        if local_video_path_s.strip():
            local_video_path_s = local_video_path_s.strip().strip('"').strip("'")
            
    num_parts_s = st.number_input("Số phần cần chia (N)", min_value=2, max_value=100, value=3, step=1, key="s_num_parts")
    fast_split_s = st.checkbox("Cắt nhanh không mã hóa lại (Fast Split)", value=True, key="s_fast_split", 
                               help="Nếu bật, FFmpeg sẽ sử dụng cơ chế stream copy (cực nhanh, không mất chất lượng). Nếu tắt, video sẽ được re-encode (chậm hơn nhưng chính xác khung hình/keyframe).")
    
    has_video_s = bool(uploaded_video_s) if input_method_s == "Upload qua trình duyệt" else bool(local_video_path_s)
    
    st.info(f"📁 Thư mục lưu kết quả mặc định: `{st.session_state['output_folder']}` (Bạn có thể thay đổi thư mục này ở ô cấu hình đầu trang)")
    
    if st.button("Bắt đầu chia Video", type="primary", key="s_run_btn"):
        if not has_video_s:
            st.error("Vui lòng tải lên video hoặc nhập đường dẫn video cục bộ!")
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
            
            result = {"parts": None, "error": None, "done": False, "user_output_dir": None}
            
            def run_in_thread():
                try:
                    import importlib
                    import app.services.composer
                    importlib.reload(app.services.composer)
                    from app.services.composer import composer
                    
                    logger.info("=== Bắt đầu Workflow 3 (Split Video) ===")
                    if input_method_s == "Upload qua trình duyệt":
                        video_path = save_uploaded_file(uploaded_video_s, task_dir)
                    else:
                        video_path = local_video_path_s
                        if not os.path.exists(video_path):
                            raise FileNotFoundError(f"Không tìm thấy file video cục bộ: {video_path}")
                    
                    output_files = composer.split_video_into_parts(
                        task_id=task_id,
                        video_path=video_path,
                        num_parts=int(num_parts_s),
                        fast_split=fast_split_s
                    )
                    
                    # Copy split parts to the user-specified output folder for easy access
                    base_name, _ = os.path.splitext(os.path.basename(video_path))
                    out_dir_name = f"split_{base_name}_{int(time.time())}"
                    user_output_dir = os.path.join(st.session_state["output_folder"], out_dir_name)
                    os.makedirs(user_output_dir, exist_ok=True)
                    
                    copied_files = []
                    for f_path in output_files:
                        dest_f = os.path.join(user_output_dir, os.path.basename(f_path))
                        shutil.copy(f_path, dest_f)
                        copied_files.append(dest_f)
                    
                    # Also copy the ZIP file
                    zip_src = os.path.join(task_dir, "split_parts.zip")
                    if os.path.exists(zip_src):
                        shutil.copy(zip_src, os.path.join(user_output_dir, "split_parts.zip"))
                        
                    result["parts"] = output_files
                    result["user_output_dir"] = user_output_dir
                    logger.info(f"Đã sao chép các phần video sang thư mục đầu ra: {user_output_dir}")
                    logger.info("=== Hoàn thành Workflow 3 thành công! ===")
                except Exception as ex:
                    import traceback
                    logger.error(f"Lỗi: {ex}")
                    logger.error(traceback.format_exc())
                    result["error"] = ex
                finally:
                    result["done"] = True
                    try:
                        logger.remove(sink_id)
                    except Exception:
                        pass
            
            thread = threading.Thread(target=run_in_thread, name=f"task_{task_id}")
            thread.start()
            
            # Read and display log in real-time
            while not result["done"]:
                time.sleep(0.5)
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        log_data = f.read()
                    try:
                        log_placeholder.code(log_data, language="text")
                    except Exception:
                        break
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                try:
                    log_placeholder.code(log_data, language="text")
                except Exception:
                    pass
            
            if result["error"]:
                st.error(f"Lỗi khi chia video: {result['error']}")
            elif result["parts"]:
                parts = result["parts"]
                user_output_dir = result["user_output_dir"]
                st.success(f"🎉 Đã chia video thành {len(parts)} phần thành công!")
                st.info(f"📁 Các phần video đã được lưu tại: `{user_output_dir}`")
                
                # Zip file download
                zip_path = os.path.join(task_dir, "split_parts.zip")
                if os.path.exists(zip_path):
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="📥 Tải xuống tất cả các phần (ZIP)",
                            data=f,
                            file_name="split_parts.zip",
                            mime="application/zip",
                            key="dl_zip_immediate_workflow3"
                        )
                
                for idx, part_path in enumerate(parts):
                    part_filename = os.path.basename(part_path)
                    st.markdown(f"**Phần {idx+1}: {part_filename}**")
                    st.video(part_path)
                    with open(part_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Tải xuống phần {idx+1}",
                            data=f,
                            file_name=part_filename,
                            mime="video/mp4",
                            key=f"dl_part_immediate_workflow3_{idx}"
                        )

