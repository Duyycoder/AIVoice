import os
import sys
import tempfile
# Force-load DLLs in exact order: torch -> faster_whisper -> cv2 to prevent Windows C++ CUDA/OpenMP abort
try:
    import torch  # noqa: F401
    import faster_whisper  # noqa: F401
    import cv2  # noqa: F401
except Exception:
    pass

# ── PyTorch / Torchao compatibility patches ──────────────────────────────────
# torchao is incompatible with PyTorch 2.6. These patches run before any import
# of torchao or transformers to handle compatibility and reload issues.

# Clean pre-existing monkey patches and cached failed imports from memory is no longer needed
# as we have installed compatible library versions.


# 1. register_constant patch
try:
    import torch.utils._pytree
    _orig_reg = getattr(torch.utils._pytree, "register_constant", None)
    if _orig_reg is None or not hasattr(_orig_reg, "__is_patched__"):
        def _safe_register_constant(cls):
            return cls
        _safe_register_constant.__is_patched__ = True
        torch.utils._pytree.register_constant = _safe_register_constant
except Exception:
    pass

# 2. Library.define, Library.impl, _add_op_to_registry, and register_fake patches
try:
    import torch.library
    # Store true original methods if not already done
    if not hasattr(torch.library, "_true_original_define"):
        torch.library._true_original_define = torch.library.Library.define
    if not hasattr(torch.library, "_true_original_impl"):
        torch.library._true_original_impl = torch.library.Library.impl
    if not hasattr(torch.library, "_true_original_register_fake"):
        torch.library._true_original_register_fake = torch.library.register_fake

    import torch._decomp
    if not hasattr(torch._decomp, "_true_original_add_op"):
        torch._decomp._true_original_add_op = torch._decomp._add_op_to_registry

    # Define safe wrappers that delegate directly to the true originals
    def _safe_define(self, *args, **kwargs):
        try:
            return torch.library._true_original_define(self, *args, **kwargs)
        except RuntimeError as e:
            if "multiple times" in str(e):
                return None
            raise
    _safe_define.__is_patched__ = True
    torch.library.Library.define = _safe_define

    def _safe_impl(self, *args, **kwargs):
        try:
            return torch.library._true_original_impl(self, *args, **kwargs)
        except RuntimeError as e:
            if "already a kernel registered" in str(e):
                return None
            raise
    _safe_impl.__is_patched__ = True
    torch.library.Library.impl = _safe_impl

    def _safe_add_op(registry, op, fn):
        try:
            return torch._decomp._true_original_add_op(registry, op, fn)
        except RuntimeError as e:
            if "duplicate registrations" in str(e):
                return None
            raise
    _safe_add_op.__is_patched__ = True
    torch._decomp._add_op_to_registry = _safe_add_op

    def _safe_register_fake(op, func=None, *args, **kwargs):
        if func is not None:
            try:
                return torch.library._true_original_register_fake(op, func, *args, **kwargs)
            except RuntimeError as e:
                if "already has an fake impl registered" in str(e):
                    return func
                raise
        decorator = torch.library._true_original_register_fake(op, None, *args, **kwargs)
        def safe_decorator(f):
            try:
                return decorator(f)
            except RuntimeError as e:
                if "already has an fake impl registered" in str(e):
                    return f
                raise
        return safe_decorator
    _safe_register_fake.__is_patched__ = True
    torch.library.register_fake = _safe_register_fake
except Exception:
    pass
except Exception:
    pass

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
                        log_placeholder.text(log_data)
                    except Exception:
                        break
            st.rerun()
        else:
            if os.path.exists(task_info["log_path"]):
                with open(task_info["log_path"], "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                log_placeholder.text(log_data)
                
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
    providers = ["OpenAI", "Google Gemini", "Local Gemini"]
    current_provider = config.app.get("llm_provider", "OpenAI")
    if current_provider not in providers:
        current_provider = "OpenAI"
    llm_provider = st.selectbox("LLM Provider", providers, index=providers.index(current_provider))
    
    # Configure defaults based on provider
    if llm_provider == "OpenAI":
        api_key_label = "OpenAI API Key"
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
    elif llm_provider == "Google Gemini":
        api_key_label = "Gemini API Key"
        default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        default_model = "gemini-2.0-flash"
    else:  # Local Gemini
        api_key_label = "Local Gemini API Key"
        default_base_url = "http://localhost:7860/v1"
        default_model = "gemini-3-flash"
        
    api_key_val = config.app.get("openai_api_key", "")
    
    # Auto-adjust API key defaults to prevent mismatches
    if llm_provider == "Local Gemini" and (not api_key_val or api_key_val.startswith("AQ.")):
        api_key_val = "sk-gemini-YrVwXWGegzkFlevHPdQy7Fpry14HJVirqvnuxukz"
    elif llm_provider == "Google Gemini" and api_key_val.startswith("sk-gemini-"):
        api_key_val = ""
        
    api_key = st.text_input(api_key_label, value=api_key_val, type="password")
    
    # Configure model name
    model_val = config.app.get("openai_model", default_model)
    if llm_provider != config.app.get("llm_provider"):
        model_val = default_model
    if llm_provider == "Google Gemini" and model_val == "gemini-1.5-flash":
        model_val = "gemini-2.0-flash"
    elif llm_provider == "Local Gemini" and model_val not in ["gemini-3-flash", "gemini-3-flash-thinking", "gemini-3-pro"]:
        model_val = "gemini-3-flash"
        
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Manual Mode (Upload)", 
    "Auto Mode (Fetch)", 
    "Split Video (Workflow 3)", 
    "Auto Translate & Sub (Workflow 4)",
    "📖 AI Storytelling 2D"
])

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
                        log_placeholder.text(log_data)
                    except Exception:
                        break
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                try:
                    log_placeholder.text(log_data)
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

with tab5:
    st.header("📖 AI Storytelling 2D")
    st.subheader("📚 CONTEXT WINDOW")
    
    from app.services.storytelling.context_manager import ContextManager
    import os
    
    all_contexts = ContextManager.list_all_contexts()
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_story = st.selectbox("Bộ truyện", ["Tạo mới..."] + all_contexts, key="selected_story_slug")
        
    with col2:
        if st.button("➕ Tạo mới"):
            st.session_state.create_context_modal = True
            
    if selected_story == "Tạo mới..." or st.session_state.get("create_context_modal"):
        with st.form("new_context_form"):
            st.write("Tạo bộ truyện mới")
            new_story_name = st.text_input("Tên bộ truyện", value="Người Trên Vạn Người")
            new_story_slug = st.text_input("Slug (không dấu, cách nhau bởi _)", value="Nguoi_Tren_Van_Nguoi")
            new_genre = st.text_input("Thể loại", value="xianxia")
            submitted = st.form_submit_button("Tạo")
            if submitted:
                if new_story_slug:
                    ctx_mgr = ContextManager(new_story_slug)
                    ctx_mgr.create_context(new_story_name, new_genre)
                    st.success("Tạo thành công!")
                    st.session_state.create_context_modal = False
                    st.session_state.selected_story_slug = new_story_slug
                    st.rerun()
                    
    elif selected_story and selected_story != "Tạo mới...":
        ctx_mgr = ContextManager(selected_story)
        ctx = ctx_mgr.load_context()
        
        st.write("Quản lý nhân vật:")
        chars = ctx_mgr.list_characters()
        char_options = ["[+] Thêm nhân vật mới"] + [c.name for c in chars]
        
        # Determine defaults
        selected_char_name = st.selectbox("Chọn nhân vật để xem/sửa", char_options)
        is_new = selected_char_name == "[+] Thêm nhân vật mới"
        
        def_name = "" if is_new else selected_char_name
        def_desc = ""
        def_kw = ""
        selected_slug = ""
        has_face = False
        
        if not is_new:
            for c in chars:
                if c.name == selected_char_name:
                    def_desc = c.description
                    def_kw = c.keywords_en
                    selected_slug = c.slug
                    has_face = c.has_embedding
                    break
        
        status_text = "Trạng thái Face Embedding: " + ("✅ Đã có" if has_face else "⚠️ Chưa có") if not is_new else ""
        if status_text:
            st.info(status_text)
            
        with st.form("char_form"):
            char_name = st.text_input("Tên nhân vật", value=def_name)
            char_desc = st.text_input("Mô tả ngoại hình (cho LLM)", value=def_desc)
            char_keywords = st.text_input("Keywords (cho SD)", value=def_kw)
            ref_img = st.file_uploader("Upload ảnh chân dung (ghi đè ảnh cũ nếu có)", type=["png", "jpg", "jpeg"])
            
            col_save, col_del = st.columns([1, 1])
            with col_save:
                submit_save = st.form_submit_button("💾 Lưu nhân vật")
            with col_del:
                submit_del = st.form_submit_button("🗑️ Xóa nhân vật")
                
            if submit_save:
                if not char_name:
                    st.error("Tên không được để trống!")
                else:
                    ref_path = ""
                    if ref_img:
                        ref_dir = os.path.join("storage", "contexts", selected_story, "temp")
                        os.makedirs(ref_dir, exist_ok=True)
                        ref_path = os.path.join(ref_dir, ref_img.name)
                        with open(ref_path, "wb") as f:
                            f.write(ref_img.getbuffer())
                    
                    ctx_mgr.add_character(char_name, char_desc, char_keywords, ref_path)
                    
                    if ref_path:
                        try:
                            from app.services.storytelling.face_extractor import extract_and_save_face_embedding
                            from app.services.storytelling.hardware_adapter import get_hardware_config
                            import re
                            slug = re.sub(r'[^a-zA-Z0-9]+', '_', char_name.lower()).strip('_')
                            emb_path = os.path.join(ctx_mgr.chars_dir, slug, "face.ipadpt")
                            hw_config = get_hardware_config()
                            extract_and_save_face_embedding(ref_path, emb_path, device=hw_config["face_device"])
                            
                            char = ctx_mgr.get_character(slug)
                            if char:
                                char.has_embedding = True
                                ctx_mgr.save_context()
                            st.success("Đã lưu nhân vật và trích xuất Face Embedding thành công!")
                        except Exception as e:
                            st.error(f"Lỗi extract face: {e}")
                    else:
                        st.success("Đã lưu thông tin nhân vật!")
                    
                    st.rerun()
                    
            if submit_del:
                if is_new:
                    st.warning("Chưa chọn nhân vật để xóa!")
                else:
                    success = ctx_mgr.delete_character(selected_slug)
                    if success:
                        st.success(f"Đã xóa nhân vật {selected_char_name}")
                        st.rerun()
                    else:
                        st.error("Xóa thất bại!")

        st.markdown("---")
        
        st.subheader("📚 BÓC TÁCH NHÂN VẬT TỰ ĐỘNG")
        st.write("Tải lên các chương truyện để tự động bóc tách nhân vật và tìm kiếm ngoại hình trên mạng.")
        
        if "pending_characters" in st.session_state and st.session_state.pending_characters:
            if st.session_state.get("pending_story_slug") != selected_story:
                del st.session_state.pending_characters
                st.rerun()
            else:
                st.info("📋 BẠN CÓ NHÂN VẬT CHƯA LƯU. VUI LÒNG DUYỆT VÀ LƯU BÊN DƯỚI!")
                with st.form("confirm_chars_form"):
                    st.write("Vui lòng kiểm tra lại thông tin và hình ảnh nhân vật trước khi lưu.")
                    saved_chars_count = 0
                    
                    for idx, char in enumerate(st.session_state.pending_characters):
                        st.markdown(f"### 👤 {char['name']}")
                        save_this = st.checkbox(f"Lưu {char['name']}", value=True, key=f"save_char_{idx}")
                        
                        c_name = st.text_input("Tên nhân vật", value=char['name'], key=f"name_char_{idx}")
                        c_desc = st.text_area("Mô tả ngoại hình (tiếng Việt)", value=char['description'], key=f"desc_char_{idx}")
                        c_kw = st.text_input("Keywords (Stable Diffusion - tiếng Anh)", value=char['keywords_en'], key=f"kw_char_{idx}")
                        
                        image_urls = char.get("image_urls", [])
                        selected_img_url = ""
                        if image_urls:
                            st.write("Ảnh chân dung từ internet:")
                            img_opts = ["Không dùng ảnh mạng (tải lên ảnh riêng sau)"] + image_urls
                            cols = st.columns(min(len(image_urls), 4))
                            for img_idx, url in enumerate(image_urls[:4]):
                                with cols[img_idx]:
                                    st.image(url, use_container_width=True)
                            selected_img_url = st.radio("Chọn ảnh dùng làm Face Embedding:", img_opts, key=f"img_select_{idx}")
                        else:
                            st.info("Không tìm thấy ảnh trên mạng.")
                            
                        char["final_name"] = c_name
                        char["final_description"] = c_desc
                        char["final_keywords_en"] = c_kw
                        char["final_image_url"] = selected_img_url if selected_img_url != "Không dùng ảnh mạng (tải lên ảnh riêng sau)" else ""
                        char["should_save"] = save_this
                        st.markdown("---")
                        
                    col_ok, col_cancel = st.columns([1, 1])
                    with col_ok:
                        submitted_confirm = st.form_submit_button("💾 Xác nhận lưu các nhân vật")
                    with col_cancel:
                        submitted_cancel = st.form_submit_button("❌ Hủy bỏ kết quả")
                        
                    if submitted_confirm:
                        with st.spinner("Đang lưu nhân vật và tải ảnh..."):
                            import tempfile
                            import requests
                            for char in st.session_state.pending_characters:
                                if char["should_save"]:
                                    name = char["final_name"]
                                    desc = char["final_description"]
                                    kw = char["final_keywords_en"]
                                    img_url = char["final_image_url"]
                                    
                                    ref_path = ""
                                    if img_url:
                                        try:
                                            suffix = ".jpg"
                                            if ".png" in img_url.lower(): suffix = ".png"
                                            elif ".webp" in img_url.lower(): suffix = ".webp"
                                            r = requests.get(img_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                                            if r.status_code == 200:
                                                temp_dir = os.path.join("storage", "contexts", selected_story, "temp")
                                                os.makedirs(temp_dir, exist_ok=True)
                                                import re
                                                char_slug = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_')
                                                ref_path = os.path.join(temp_dir, f"{char_slug}_web{suffix}")
                                                with open(ref_path, "wb") as f:
                                                    f.write(r.content)
                                        except Exception as e:
                                            st.error(f"Lỗi tải ảnh {name}: {e}")
                                            ref_path = ""
                                            
                                    ctx_mgr.add_character(name, desc, kw, ref_path)
                                    if ref_path and os.path.exists(ref_path):
                                        try:
                                            from app.services.storytelling.face_extractor import extract_and_save_face_embedding
                                            from app.services.storytelling.hardware_adapter import get_hardware_config
                                            import re
                                            slug = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_')
                                            emb_path = os.path.join(ctx_mgr.chars_dir, slug, "face.ipadpt")
                                            hw_config = get_hardware_config()
                                            extract_and_save_face_embedding(ref_path, emb_path, device=hw_config["face_device"])
                                            
                                            char_obj = ctx_mgr.get_character(slug)
                                            if char_obj:
                                                char_obj.has_embedding = True
                                                ctx_mgr.save_context()
                                        except Exception as e:
                                            st.warning(f"Lỗi face embedding {name}: {e}")
                                            
                                    saved_chars_count += 1
                                    
                            st.success(f"Đã lưu thành công {saved_chars_count} nhân vật!")
                            del st.session_state.pending_characters
                            st.rerun()
                            
                    if submitted_cancel:
                        del st.session_state.pending_characters
                        st.info("Đã hủy kết quả bóc tách.")
                        st.rerun()
        
        if "pending_characters" not in st.session_state or not st.session_state.pending_characters:
            with st.form("extract_chars_form"):
                uploaded_chapters = st.file_uploader("Tải lên chương truyện (.txt, .md)", type=["txt", "md"], accept_multiple_files=True)
                enable_web_search = st.checkbox("Tìm kiếm thông tin & hình ảnh trên mạng", value=True)
                submit_extract = st.form_submit_button("🔍 Bắt đầu bóc tách")
                
                if submit_extract:
                    if not uploaded_chapters:
                        st.warning("Vui lòng tải lên ít nhất một file chương truyện.")
                    else:
                        chapter_texts = []
                        for f in uploaded_chapters:
                            content = f.read().decode("utf-8", errors="ignore")
                            chapter_texts.append(content)
                            
                        with st.spinner("Đang bóc tách nhân vật bằng LLM (có thể mất 10-30s)..."):
                            from app.services.storytelling.character_extractor import process_chapters_and_extract_characters
                            extracted_chars = process_chapters_and_extract_characters(
                                chapter_texts=chapter_texts,
                                story_name=ctx.story_name,
                                genre=ctx.genre,
                                enable_web_search=enable_web_search
                            )
                            if extracted_chars:
                                st.session_state.pending_characters = extracted_chars
                                st.session_state.pending_story_slug = selected_story
                                st.rerun()
                            else:
                                st.warning("Không phát hiện nhân vật nào trong văn bản tải lên.")

        st.markdown("---")
        st.subheader("⚙️ CẤU HÌNH PHẦN CỨNG & TĂNG TỐC")
        
        from app.services.storytelling.hardware_adapter import get_hardware_profile, get_hardware_config
        current_hw_profile = get_hardware_profile()
        hw_profile_opts = ["auto", "cuda_high", "cuda_low", "cpu"]
        hw_profile_labels = [
            "🤖 Tự động tối ưu (Auto-Detect)",
            "⚡ Tăng tốc tối đa (GPU >= 8GB VRAM - e.g. RTX 5060)",
            "💾 Tiết kiệm VRAM (GPU <= 6GB VRAM)",
            "💻 Chỉ chạy CPU (CPU Only)"
        ]
        
        if current_hw_profile not in hw_profile_opts:
            current_hw_profile = "auto"
            
        selected_hw_label = st.selectbox(
            "Chế độ chạy mô hình AI:",
            hw_profile_labels,
            index=hw_profile_opts.index(current_hw_profile),
            help="Tự động tối ưu sẽ nhận diện card đồ hoạ và dung lượng VRAM thực tế của máy bạn để đưa ra cấu hình tối ưu nhất."
        )
        selected_hw_profile = hw_profile_opts[hw_profile_labels.index(selected_hw_label)]
        
        if selected_hw_profile != current_hw_profile:
            config.storytelling["hardware_profile"] = selected_hw_profile
            config.save_config()
            st.success(f"Đã cập nhật cấu hình phần cứng thành: {selected_hw_profile}!")
            st.rerun()
            
        hw_config_active = get_hardware_config()
        st.info(f"💡 Hồ sơ phần cứng hoạt động: **{hw_config_active['profile_name']}** (SD Device: `{hw_config_active['sd_device']}`, CPU Offload: `{hw_config_active['enable_cpu_offload']}`, Face Device: `{hw_config_active['face_device']}`).")
        
        st.markdown("---")
        st.subheader("📁 DỮ LIỆU ĐẦU VÀO & THỰC THI PIPELINE")
        
        from app.services.storytelling.orchestrator import StorytellingOrchestrator
        from app.services.storytelling.models import Scene
        orchestrator = StorytellingOrchestrator(ctx_mgr)
        current_state = orchestrator.load_state()
        
        exec_mode = st.radio(
            "⚡ Chế độ thực thi Pipeline:",
            [
                "🛑 3 Trạm Tương Tác (Human-in-the-Loop Studio)", 
                "⚡ Chạy Tự Động Toàn Bộ (Skip Checkpoints)",
                "📦 Chạy Hàng Loạt (Batch Processing)"
            ],
            index=0,
            horizontal=True
        )
        
        if exec_mode.startswith("⚡"):
            col_md, col_audio, col_srt = st.columns(3)
            with col_md:
                md_file = st.file_uploader("File kịch bản (.md)", type=["md"], key="auto_md")
            with col_audio:
                audio_file = st.file_uploader("File audio (.mp3, .wav)", type=["mp3", "wav"], key="auto_audio")
            with col_srt:
                srt_file = st.file_uploader("File phụ đề (.srt) - Tuỳ chọn", type=["srt"], key="auto_srt")
                
            if st.button("🚀 Chạy Tự Động Toàn Bộ Pipeline", key="btn_auto_run"):
                if not md_file or not audio_file:
                    st.error("Vui lòng tải lên kịch bản và audio!")
                else:
                    st.info("Đang khởi tạo Task...")
                    temp_dir = os.path.join("storage", "contexts", selected_story, "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    md_path = os.path.join(temp_dir, md_file.name)
                    audio_path = os.path.join(temp_dir, audio_file.name)
                    with open(md_path, "wb") as f: f.write(md_file.getbuffer())
                    with open(audio_path, "wb") as f: f.write(audio_file.getbuffer())
                    srt_path = ""
                    if srt_file:
                        srt_path = os.path.join(temp_dir, srt_file.name)
                        with open(srt_path, "wb") as f: f.write(srt_file.getbuffer())
                    
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    def update_ui(msg, pct):
                        progress_text.text(msg)
                        progress_bar.progress(pct / 100.0)
                        
                    try:
                        final_video = orchestrator.run_pipeline(md_path, audio_path, srt_path, progress_callback=update_ui)
                        st.success("🎉 Đã hoàn thành AI Storytelling Video!")
                        st.video(final_video)
                    except Exception as e:
                        st.error(f"❌ Có lỗi xảy ra trong quá trình chạy Pipeline: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        elif exec_mode.startswith("📦"):
            st.markdown("### 📦 Chạy Hàng Loạt (Batch Processing)")
            st.info("Hệ thống sẽ tự động tìm các file .md và file audio (.mp3, .wav) trong thư mục Input, sắp xếp và ghép cặp chúng theo thứ tự tên file, sau đó render hàng loạt.")
            
            batch_input_dir = st.text_input("📁 Thư mục Input (chứa .md và audio)", placeholder="VD: D:\\Projects\\AudioBooks\\Chuong1_10")
            batch_output_dir = st.text_input("💾 Thư mục Output (nơi lưu video mp4)", placeholder="VD: D:\\Projects\\AudioBooks\\Output")
            
            if batch_input_dir and os.path.isdir(batch_input_dir):
                import glob
                import re
                
                def natural_keys(text):
                    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
                
                md_files = glob.glob(os.path.join(batch_input_dir, "*.md"))
                md_files.sort(key=natural_keys)
                
                audio_files = [f for f in glob.glob(os.path.join(batch_input_dir, "*.*")) if f.endswith(('.mp3', '.wav'))]
                audio_files.sort(key=natural_keys)
                
                st.write(f"Tìm thấy **{len(md_files)}** file kịch bản và **{len(audio_files)}** file audio.")
                
                pairs = []
                for i in range(min(len(md_files), len(audio_files))):
                    pairs.append({
                        "md": md_files[i],
                        "audio": audio_files[i]
                    })
                
                if pairs:
                    st.markdown("#### 📋 Preview Ghép Cặp (Natural Sort)")
                    preview_data = [{"Kịch Bản (.md)": os.path.basename(p["md"]), "Âm Thanh": os.path.basename(p["audio"])} for p in pairs]
                    st.table(preview_data)
                    
                    if batch_output_dir and st.button("🚀 Bắt Đầu Chạy Hàng Loạt", key="btn_run_batch"):
                        os.makedirs(batch_output_dir, exist_ok=True)
                        st.info("Đang xử lý Batch... Vui lòng không đóng trình duyệt!")
                        
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        success_count = 0
                        import shutil
                        
                        for idx, pair in enumerate(pairs):
                            md_path = pair["md"]
                            audio_path = pair["audio"]
                            base_name = os.path.splitext(os.path.basename(md_path))[0]
                            
                            progress_text.text(f"Đang xử lý video {idx+1}/{len(pairs)}: {base_name}...")
                            progress_bar.progress(idx / len(pairs))
                            
                            try:
                                # Tạo temp dir riêng cho mỗi task để tránh trùng lặp state
                                temp_dir = os.path.join("storage", "contexts", selected_story, "temp", f"batch_{idx}")
                                os.makedirs(temp_dir, exist_ok=True)
                                
                                temp_md = os.path.join(temp_dir, os.path.basename(md_path))
                                temp_audio = os.path.join(temp_dir, os.path.basename(audio_path))
                                shutil.copy2(md_path, temp_md)
                                shutil.copy2(audio_path, temp_audio)
                                
                                # Clear state cũ
                                orchestrator.clear_state()
                                
                                # Chạy pipeline
                                final_video = orchestrator.run_pipeline(temp_md, temp_audio, srt_path="", progress_callback=None)
                                
                                # Move và đổi tên sang output dir
                                out_file = os.path.join(batch_output_dir, f"{base_name}.mp4")
                                shutil.copy2(final_video, out_file)
                                success_count += 1
                                st.success(f"✅ Đã hoàn thành: {base_name}")
                                
                                # Dọn dẹp rác (nếu cần)
                                try: shutil.rmtree(temp_dir)
                                except: pass
                                
                            except Exception as e:
                                st.error(f"❌ Lỗi khi xử lý {base_name}: {e}")
                                logger.error(f"Batch Error on {base_name}: {e}")
                                
                        progress_bar.progress(1.0)
                        progress_text.text(f"Đã hoàn thành toàn bộ Batch! ({success_count}/{len(pairs)} thành công)")
                        st.balloons()
        elif exec_mode.startswith("🛑"):
            st.markdown("### 🎙️ Human-in-the-Loop Studio")
            
            step_status = current_state.get("step", "INIT") if current_state else "INIT"
            
            col_stat1, col_stat2 = st.columns([3, 1])
            with col_stat1:
                st.info(f"📍 **Trạng thái hiện tại:** `{step_status}`")
            with col_stat2:
                if current_state and st.button("🗑️ Reset Trạng Thái / Làm Lại", key="btn_reset_state"):
                    orchestrator.clear_state()
                    st.rerun()
                    
            if step_status == "INIT":
                st.markdown("#### 🚩 Trạm 1: Upload Dữ Liệu & Phân Tách Kịch Bản")
                col_md, col_audio, col_srt = st.columns(3)
                with col_md:
                    md_file = st.file_uploader("File kịch bản (.md)", type=["md"], key="st1_md")
                with col_audio:
                    audio_file = st.file_uploader("File audio (.mp3, .wav)", type=["mp3", "wav"], key="st1_audio")
                with col_srt:
                    srt_file = st.file_uploader("File phụ đề (.srt) - Tuỳ chọn", type=["srt"], key="st1_srt")
                
                use_whisper = st.checkbox("Sử dụng AI Whisper để dịch Audio (Khuyên dùng cho video dài, Tốn VRAM)", value=True)
                    
                if st.button("▶ Chạy Trạm 1: Tạo Script & Prompt LLM", key="btn_run_st1"):
                    if not md_file or not audio_file:
                        st.error("Vui lòng tải lên kịch bản và audio!")
                    else:
                        temp_dir = os.path.join("storage", "contexts", selected_story, "temp")
                        os.makedirs(temp_dir, exist_ok=True)
                        md_path = os.path.join(temp_dir, md_file.name)
                        audio_path = os.path.join(temp_dir, audio_file.name)
                        with open(md_path, "wb") as f: f.write(md_file.getbuffer())
                        with open(audio_path, "wb") as f: f.write(audio_file.getbuffer())
                        srt_path = ""
                        if srt_file:
                            srt_path = os.path.join(temp_dir, srt_file.name)
                            with open(srt_path, "wb") as f: f.write(srt_file.getbuffer())
                            
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        def update_ui(msg, pct):
                            progress_text.text(msg)
                            progress_bar.progress(pct / 100.0)
                            
                        try:
                            orchestrator.step1_generate_script(
                                md_path, audio_path, srt_path, 
                                progress_callback=update_ui, 
                                use_whisper=use_whisper
                            )
                            st.success("✅ Trạm 1 hoàn thành! Đang chuyển sang Script Studio...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi Trạm 1: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            
            elif step_status == "SCRIPT_READY":
                st.markdown("#### 📝 Trạm 1 — Script Studio: Chỉnh Sửa Lời Thoại & Prompt")
                scenes_data = current_state.get("scenes", [])
                
                import pandas as pd
                df_scenes = pd.DataFrame([
                    {
                        "ID": s["scene_id"],
                        "Bắt đầu (s)": round(s["start_time"], 2),
                        "Kết thúc (s)": round(s["end_time"], 2),
                        "Lời thoại (VI)": s["text_vi"],
                        "Prompt (EN)": s["image_prompt"]
                    } for s in scenes_data
                ])
                
                st.caption("💡 Bạn có thể nhấp cú đúp vào bất kỳ ô nào để chỉnh sửa trực tiếp Lời thoại hoặc Prompt tiếng Anh trước khi AI vẽ hình.")
                edited_df = st.data_editor(df_scenes, use_container_width=True, num_rows="fixed", key="script_editor")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💾 Lưu Thay Đổi Kịch Bản", key="btn_save_script"):
                        for idx, row in edited_df.iterrows():
                            scenes_data[idx]["text_vi"] = row["Lời thoại (VI)"]
                            scenes_data[idx]["image_prompt"] = row["Prompt (EN)"]
                        scenes_obj = [Scene(**s) for s in scenes_data]
                        orchestrator.save_state("SCRIPT_READY", scenes_obj, current_state["task_dir"], current_state.get("audio_path", ""), current_state.get("srt_path", ""), current_state.get("md_path", ""))
                        st.success("Đã lưu chỉnh sửa kịch bản!")
                with col_btn2:
                    if st.button("▶ Xác Nhận & Sang Trạm 2 (Sinh Ảnh SD)", key="btn_to_st2", type="primary"):
                        for idx, row in edited_df.iterrows():
                            scenes_data[idx]["text_vi"] = row["Lời thoại (VI)"]
                            scenes_data[idx]["image_prompt"] = row["Prompt (EN)"]
                        scenes_obj = [Scene(**s) for s in scenes_data]
                        orchestrator.save_state("SCRIPT_READY", scenes_obj, current_state["task_dir"], current_state.get("audio_path", ""), current_state.get("srt_path", ""), current_state.get("md_path", ""))
                        
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        def update_ui(msg, pct):
                            progress_text.text(msg)
                            progress_bar.progress(pct / 100.0)
                        try:
                            orchestrator.step2_generate_images(scenes_obj, current_state["task_dir"], progress_callback=update_ui)
                            st.success("✅ Trạm 2 hoàn thành! Đang chuyển sang Storyboard Studio...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi Trạm 2: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            
            elif step_status == "STORYBOARD_READY":
                st.markdown("#### 🖼️ Trạm 2 — Storyboard Studio: Kiểm Duyệt & Re-roll Ảnh")
                scenes_data = current_state.get("scenes", [])
                
                st.write("Dưới đây là storyboard các cảnh đã vẽ. Bạn có thể Re-roll từng ảnh hoặc tự tải lên ảnh ngoại tuyến thay thế.")
                
                cols_per_row = 3
                grid_cols = st.columns(cols_per_row)
                for idx, s_dict in enumerate(scenes_data):
                    col = grid_cols[idx % cols_per_row]
                    with col:
                        st.markdown(f"**Cảnh #{s_dict['scene_id']+1}** (`{s_dict['start_time']:.1f}s - {s_dict['end_time']:.1f}s`)")
                        img_path = s_dict.get("frame_path", "")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.warning("Chưa có ảnh hoặc mất file")
                        st.caption(f"🌱 Seed: `{s_dict.get('accepted_seed', -1)}`")
                        with st.expander("📝 Prompt Cảnh #" + str(s_dict['scene_id']+1)):
                            new_p = st.text_area("Sửa prompt nếu cần:", value=s_dict["image_prompt"], key=f"p_edit_{idx}")
                            
                        c_rr, c_up = st.columns([1, 1])
                        with c_rr:
                            if st.button("🔄 Re-roll", key=f"rr_{idx}"):
                                with st.spinner(f"Đang vẽ lại Cảnh #{idx+1}..."):
                                    try:
                                        orchestrator.reroll_scene(idx, new_seed=-1, new_prompt=new_p)
                                        st.success(f"Đã re-roll Cảnh #{idx+1}!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Lỗi: {e}")
                        with c_up:
                            up_img = st.file_uploader("📤 Upload", type=["png", "jpg", "jpeg"], key=f"up_{idx}", label_visibility="collapsed")
                            if up_img is not None:
                                try:
                                    # Guard: if frame_path is empty (not yet generated), save to task_dir
                                    save_path = img_path if img_path else os.path.join(
                                        current_state.get("task_dir", "storage/tasks"), "draft_frames", f"scene_{idx:03d}.png"
                                    )
                                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                    with open(save_path, "wb") as f:
                                        f.write(up_img.getbuffer())
                                    # Update state so frame_path reflects upload
                                    scenes_data[idx]["frame_path"] = save_path
                                    from app.services.storytelling.models import Scene as SceneModel
                                    scenes_obj_upd = [SceneModel(**s) for s in scenes_data]
                                    orchestrator.save_state(current_state["step"], scenes_obj_upd, current_state["task_dir"], current_state.get("audio_path", ""), current_state.get("srt_path", ""), current_state.get("md_path", ""))
                                    st.success(f"Đã thay ảnh Cảnh #{idx+1}!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Lỗi lưu ảnh: {e}")

                                    
                st.markdown("---")
                if st.button("▶ Xác Nhận Storyboard & Sang Trạm 3 (Render Final)", key="btn_to_st3", type="primary"):
                    scenes_obj = [Scene(**s) for s in scenes_data]
                    orchestrator.save_state("RENDER_READY", scenes_obj, current_state["task_dir"], current_state.get("audio_path", ""), current_state.get("srt_path", ""), current_state.get("md_path", ""))
                    st.rerun()
                    
            elif step_status in ["RENDER_READY", "DONE"]:
                st.markdown("#### 🎬 Trạm 3 — Render Studio: Hậu Kỳ Upscale & Nhạc Nền")
                scenes_data = current_state.get("scenes", [])
                scenes_obj = [Scene(**s) for s in scenes_data]
                
                from app.services.model_downloader import ensure_models_ready
                mod_status = ensure_models_ready(download_if_missing=False)
                realesrgan_ready = mod_status.get("realesrgan") == "ready"
                
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    en_upscale = st.toggle("✨ Bật AI Upscaler (RealESRGAN 4x)", value=realesrgan_ready)
                    if en_upscale and not realesrgan_ready:
                        st.warning("⚠️ Model RealESRGAN chưa được tải! Hệ thống sẽ dùng bộ lọc LANCZOS chất lượng cao thay thế hoặc tự động tải khi render.")
                with col_opt2:
                    bgm_file = st.file_uploader("🎵 Nhạc nền BGM (.mp3, .wav) - Tuỳ chọn", type=["mp3", "wav"], key="st3_bgm")
                    bgm_vol = st.slider("Âm lượng BGM", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key="st3_vol")
                    burn_subtitles = st.checkbox("Gắn cứng phụ đề (Subtitles) vào Video", value=True)
                    
                if st.button("🎬 Xuất Video Final Ngay", key="btn_run_st3", type="primary"):
                    bgm_path = ""
                    if bgm_file:
                        bgm_path = os.path.join(current_state["task_dir"], bgm_file.name)
                        with open(bgm_path, "wb") as f: f.write(bgm_file.getbuffer())
                        
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    def update_ui(msg, pct):
                        progress_text.text(msg)
                        progress_bar.progress(pct / 100.0)
                        
                    try:
                        final_vid = orchestrator.step3_render_final(
                            scenes=scenes_obj,
                            task_dir=current_state["task_dir"],
                            audio_path=current_state.get("audio_path", ""),
                            srt_path=current_state.get("srt_path", ""),
                            bgm_path=bgm_path,
                            bgm_volume=bgm_vol,
                            enable_upscaling=en_upscale,
                            burn_subtitles=burn_subtitles,
                            progress_callback=update_ui
                        )
                        st.success("🎉 CHÚC MỪNG! Video Final đã xuất thành công!")
                        st.video(final_vid)
                    except Exception as e:
                        st.error(f"❌ Lỗi Trạm 3: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        
                if step_status == "DONE":
                    final_vid_check = os.path.join(current_state["task_dir"], "final_video.mp4")
                    if os.path.exists(final_vid_check):
                        st.markdown("---")
                        st.subheader("📺 Video Hoàn Chỉnh Trước Đó:")
                        st.video(final_vid_check)

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
                        log_placeholder.text(log_data)
                    except Exception:
                        break
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                try:
                    log_placeholder.text(log_data)
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
            output_folder_captured = st.session_state["output_folder"]
            
            def run_in_thread():
                try:
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
                    user_output_dir = os.path.join(output_folder_captured, out_dir_name)
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
                        log_placeholder.text(log_data)
                    except Exception:
                        break
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                try:
                    log_placeholder.text(log_data)
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


with tab4:
    st.header("Workflow 4: Auto Translate & Sub")
    st.write("Dịch phụ đề tự động từ tiếng Anh/Trung sang tiếng Việt và chèn vào video.")
    
    input_method_t = st.radio(
        "Phương thức chọn file (Translation)",
        ["Upload qua trình duyệt", "Nhập đường dẫn cục bộ (Local Paths)"],
        horizontal=True,
        key="t_input_method"
    )
    
    uploaded_video_t = None
    local_video_path_t = ""
    
    if input_method_t == "Upload qua trình duyệt":
        uploaded_video_t = st.file_uploader("Upload Video (Bắt buộc)", type=["mp4", "mov", "avi", "mkv"], key="t_uploaded_video")
    else:
        local_video_path_t = st.text_input("Đường dẫn file Video cục bộ (Bắt buộc)", key="t_local_video", help="Ví dụ: D:/videos/english_lecture.mp4")
        if local_video_path_t.strip():
            local_video_path_t = local_video_path_t.strip().strip('"').strip("'")
            
    source_lang_t = st.selectbox(
        "Ngôn ngữ gốc của Video (Source Language)",
        ["English", "Chinese"],
        index=0,
        key="t_source_lang"
    )
    
    burn_method_t = st.selectbox(
        "Phương thức chèn phụ đề (Subtitle Burning Method)",
        ["FFmpeg Subtitles Filter (Nhanh - Khuyên Dùng)", "MoviePy Compositing (Tiêu Chuẩn - Chậm hơn)"],
        index=0,
        key="t_burn_method",
        help="FFmpeg chèn phụ đề trực tiếp ở tầng phần cứng/mã hóa, rất nhanh. MoviePy tạo ảnh dựng độc lập cho từng câu chữ nên sẽ chậm hơn."
    )
    
    has_video_t = bool(uploaded_video_t) if input_method_t == "Upload qua trình duyệt" else bool(local_video_path_t)
    
    # ------------------ Voiceover Configuration ------------------
    st.write("---")
    st.subheader("Cấu hình Lồng tiếng (Voiceover Configuration)")
    enable_voiceover_t = st.checkbox("Kích hoạt lồng tiếng (Enable Voiceover)", value=False, key="t_enable_voiceover")
    
    tts_engine_t = "edge"
    tts_voice_t = ""
    auto_clone_t = False
    ducking_ratio_t = 90.0
    
    if enable_voiceover_t:
        tts_engine_t = st.selectbox(
            "Chọn Động cơ lồng tiếng (TTS Engine)",
            ["edge", "piper", "kokoro", "vieneu", "clone"],
            index=0,
            key="t_tts_engine"
        )
        
        # Load dynamic voice list based on selected engine
        if tts_engine_t == "edge":
            # List some standard Vietnamese voices
            tts_voice_t = st.selectbox(
                "Chọn Giọng đọc (Voice)",
                ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
                index=0,
                key="t_edge_voice"
            )
        elif tts_engine_t == "piper":
            # List available ONNX models in models/piper
            piper_models = []
            piper_dir = os.path.join(root_dir, "models", "piper")
            if os.path.exists(piper_dir):
                piper_models = [f for f in os.listdir(piper_dir) if f.endswith(".onnx")]
            if not piper_models:
                piper_models = ["vi_VN-vais1000-medium.onnx"]  # fallback
            tts_voice_t = st.selectbox(
                "Chọn Mô hình Piper (.onnx)",
                piper_models,
                index=0,
                key="t_piper_voice"
            )
        elif tts_engine_t == "kokoro":
            tts_voice_t = st.selectbox(
                "Chọn Giọng đọc Kokoro",
                [
                    "diem_trinh", "hung_thinh", "mai_linh", "mai_loan", "manh_dung", 
                    "my_yen", "ngoc_huyen", "phat_tai", "thanh_dat", "thuc_trinh", 
                    "tuan_ngoc", "storyvert", "duc_an", "duc_duy"
                ],
                index=0,
                key="t_kokoro_voice"
            )
        elif tts_engine_t == "vieneu":
            vn_voice = st.selectbox(
                "Chọn Giọng đọc VieNeu (Voice)",
                ["Ngọc Lan", "Gia Bảo", "Thái Sơn", "Đức Trí", "Mỹ Duyên", "Trúc Ly", "Xuân Vĩnh", "Trọng Hữu", "Bình An", "Ngọc Linh"],
                index=0,
                key="t_vieneu_voice_name"
            )
            vn_mode = st.selectbox(
                "Chọn Chế độ VieNeu (Mode)",
                ["v3turbo", "standard"],
                index=0,
                key="t_vieneu_mode_name"
            )
            tts_voice_t = f"{vn_voice}|{vn_mode}"
        elif tts_engine_t == "clone":
            # XTTSv2 Zero Shot cloning
            auto_clone_t = st.checkbox(
                "Tự động Clone giọng nhân vật từ Video gốc (Auto-Clone)",
                value=True,
                key="t_auto_clone",
                help="Nếu bật, hệ thống sẽ quét video gốc, tự cắt nhanh một đoạn tiếng của nhân vật khoảng 6-12 giây để làm mẫu giọng nhân bản."
            )
            if not auto_clone_t:
                # Custom reference file input
                tts_voice_t = st.text_input(
                    "Đường dẫn file âm thanh mẫu (.wav)",
                    value="",
                    key="t_custom_ref_voice",
                    help="Nhập đường dẫn cục bộ đến tệp tin .wav chứa giọng nói bạn muốn nhân bản. Ví dụ: D:/voices/sample.wav"
                )
                if tts_voice_t:
                    tts_voice_t = tts_voice_t.strip().strip('"').strip("'")
            else:
                tts_voice_t = "auto"
                
        ducking_ratio_t = st.slider(
            "Tỷ lệ dìm âm lượng gốc (Original BGM/Voice Ducking %)",
            min_value=0,
            max_value=100,
            value=90,
            step=5,
            key="t_ducking_ratio",
            help="Dìm âm lượng gốc xuống bao nhiêu phần trăm khi giọng đọc tiếng Việt nói. 90% nghĩa là nhạc nền/tiếng gốc chỉ còn 10% âm lượng."
        )
    st.write("---")
    
    st.info(f"📁 Thư mục lưu kết quả mặc định: `{st.session_state['output_folder']}` (Bạn có thể thay đổi thư mục này ở ô cấu hình đầu trang)")
    
    if st.button("Bắt đầu xử lý (Translate & Sub)", type="primary", key="t_run_btn"):
        if not has_video_t:
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
            
            result = {"video": None, "error": None, "done": False, "user_output_path": None}
            output_folder_captured = st.session_state["output_folder"]
            
            def run_in_thread():
                try:
                    from app.services.composer import composer
                    
                    logger.info("=== Bắt đầu Workflow 4 (Auto Translate & Sub) ===")
                    logger.info(f"Parameters: enable_voiceover={enable_voiceover_t}, tts_engine={tts_engine_t}, tts_voice={tts_voice_t}, ducking_ratio={ducking_ratio_t}, auto_clone={auto_clone_t}")
                    if input_method_t == "Upload qua trình duyệt":
                        video_path = save_uploaded_file(uploaded_video_t, task_dir)
                    else:
                        video_path = local_video_path_t
                        if not os.path.exists(video_path):
                            raise FileNotFoundError(f"Không tìm thấy file video cục bộ: {video_path}")
                    
                    # Determine burn method
                    method = "ffmpeg" if "FFmpeg" in burn_method_t else "moviepy"
                    
                    final_video = composer.run_translation_workflow(
                        task_id=task_id,
                        video_path=video_path,
                        source_lang=source_lang_t,
                        burn_method=method,
                        enable_voiceover=enable_voiceover_t,
                        tts_engine=tts_engine_t,
                        tts_voice=tts_voice_t,
                        ducking_ratio=float(ducking_ratio_t),
                        auto_clone=auto_clone_t
                    )
                    
                    # Copy final video to the user-specified output folder for easy access
                    base_name, ext = os.path.splitext(os.path.basename(video_path))
                    out_filename = f"{base_name}_translated_{int(time.time())}{ext}"
                    user_output_path = os.path.join(output_folder_captured, out_filename)
                    
                    shutil.copy(final_video, user_output_path)
                    
                    result["video"] = final_video
                    result["user_output_path"] = user_output_path
                    logger.info(f"Đã sao chép video kết quả sang: {user_output_path}")
                    logger.info("=== Hoàn thành Workflow 4 thành công! ===")
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
                        log_placeholder.text(log_data)
                    except Exception:
                        break
            
            # Final log update
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                try:
                    log_placeholder.text(log_data)
                except Exception:
                    pass
            
            if result["error"]:
                st.error(f"Lỗi khi xử lý video: {result['error']}")
            elif result["video"]:
                final_video = result["video"]
                user_output_path = result["user_output_path"]
                st.success(f"🎉 Đã dịch và chèn phụ đề video thành công!")
                st.info(f"📁 Video kết quả đã được lưu tại: `{user_output_path}`")
                
                # Render video and download button
                filename = os.path.basename(user_output_path)
                st.video(user_output_path)
                with open(user_output_path, "rb") as f:
                    st.download_button(
                        label="📥 Tải xuống Video phụ đề",
                        data=f,
                        file_name=filename,
                        mime="video/mp4",
                        key="dl_translated_video_immediate"
                    )

