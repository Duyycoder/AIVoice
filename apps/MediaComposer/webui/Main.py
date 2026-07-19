import os
import sys
# PHẢI set TRƯỚC KHI import bất kỳ thứ gì (huggingface_hub đọc env var này lúc import)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
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

from app.config import config, load_storytelling_config
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
            if st.button("⬅️ Quay lại danh sách", width="stretch"):
                del st.session_state["selected_task_id"]
                st.rerun()
        with col_del:
            if st.button("🗑️ Xóa sạch tác vụ này", type="primary", width="stretch"):
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
                if st.button("Xem chi tiết & Log", key=f"btn_{t['id']}", width="stretch"):
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Manual Mode (Upload)",
    "Auto Mode (Fetch)",
    "Split Video (Workflow 3)",
    "Auto Translate & Sub (Workflow 4)",
    "📖 AI Storytelling 2D",
    "🎬 Video Upscale (Workflow 6)",
    "🎨 Tạo Ảnh AI (Workflow 7)"
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
    
    if st.button("Generate / Ghép xuất (Manual)", type="primary"):
        if not has_audio and not has_video:
            st.error("Vui lòng tải lên/nhập đường dẫn ít nhất một file Audio hoặc Video/Image!")
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
            
            result = {"video": None, "audio": None, "error": None, "done": False}
            
            def run_in_thread():
                try:
                    if has_audio and not has_video:
                        logger.info("=== Bắt đầu Workflow 1 (Chỉ ghép Audio) ===")
                        if input_method_m == "Upload qua trình duyệt":
                            audio_path = merge_audio_files(audio_files_m, task_dir)
                        else:
                            audio_path = merge_audio_paths(local_audio_paths_m, task_dir)
                        result["audio"] = audio_path
                        logger.info("=== Hoàn thành ghép Audio thành công! ===")
                        
                    elif not has_audio and has_video:
                        logger.info("=== Bắt đầu Workflow 1 (Chỉ ghép Video) ===")
                        if input_method_m == "Upload qua trình duyệt":
                            video_paths = []
                            for v in video_files_m:
                                video_paths.append(save_uploaded_file(v, task_dir))
                        else:
                            import re
                            def _natural_sort_key(path):
                                return [int(part) if part.isdigit() else part.lower()
                                        for part in re.split(r'(\d+)', os.path.basename(path))]
                            video_paths = sorted(local_video_paths_m, key=_natural_sort_key)
                        
                        final_video = composer.run_workflow(
                            task_id=task_id,
                            audio_path=None,
                            video_paths=video_paths,
                            auto_fetch=False,
                            bgm_file=bgm_file if enable_bgm else "",
                            video_aspect=VideoAspect(aspect_m),
                            concat_mode=VideoConcatMode.sequential,
                            enable_subtitles=False,
                            slice_video=slice_video_m
                        )
                        result["video"] = final_video
                        logger.info("=== Hoàn thành Workflow 1 (Chỉ ghép Video) thành công! ===")
                        
                    else:
                        logger.info("=== Bắt đầu Workflow 1 (Ghép cả Audio và Video) ===")
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
                            concat_mode=VideoConcatMode.sequential,
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
                st.error(f"Error during processing: {result['error']}")
            elif result.get("audio"):
                final_audio = result["audio"]
                if output_path:
                    out_dir = os.path.dirname(os.path.abspath(output_path))
                    os.makedirs(out_dir, exist_ok=True)
                    suffix = os.path.splitext(final_audio)[1] or ".wav"
                    out_path_audio = os.path.splitext(output_path)[0] + suffix
                    shutil.copy(final_audio, out_path_audio)
                    st.success(f"Audio generated and saved to: {out_path_audio}")
                    st.audio(out_path_audio)
                else:
                    st.success("Audio generated successfully!")
                    st.audio(final_audio)
            elif result.get("video"):
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

    # FIX StreamlitAPIException: không được ghi trực tiếp vào key của widget
    # SAU khi widget đã render trong cùng lượt chạy. Bộ truyện vừa tạo được
    # gửi qua key trung gian và áp vào ĐẦU lượt rerun kế tiếp (trước selectbox).
    if "_pending_story_slug" in st.session_state:
        _pending = st.session_state.pop("_pending_story_slug")
        if _pending in all_contexts:
            st.session_state["selected_story_slug"] = _pending

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
                    # Không ghi thẳng vào key widget — gửi qua key trung gian,
                    # rerun kế tiếp sẽ áp vào selectbox trước khi nó render
                    st.session_state["_pending_story_slug"] = new_story_slug
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
            ref_imgs = st.file_uploader("Upload ảnh nhân vật (ảnh đầu = ref chính, tất cả vào dataset)",
                                        type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            
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
                    if ref_imgs:
                        ref_dir = os.path.join("storage", "contexts", selected_story, "temp")
                        os.makedirs(ref_dir, exist_ok=True)
                        # Ảnh đầu = ref chính (giữ hành vi cũ)
                        ref_path = os.path.join(ref_dir, ref_imgs[0].name)
                        with open(ref_path, "wb") as f:
                            f.write(ref_imgs[0].getbuffer())
                    
                    ctx_mgr.add_character(char_name, char_desc, char_keywords, ref_path)
                    
                    import re as _re_slug
                    _slug = _re_slug.sub(r'[^a-zA-Z0-9]+', '_', char_name.lower()).strip('_')

                    # Lưu TẤT CẢ ảnh upload vào dataset dạng approved_*
                    if ref_imgs:
                        for uploaded_file in ref_imgs:
                            try:
                                from PIL import Image as PILImage
                                import io
                                _img = PILImage.open(io.BytesIO(uploaded_file.getbuffer())).convert("RGB")
                                ctx_mgr.add_dataset_image(_slug, _img, "approved")
                            except Exception as e_ds:
                                logger.warning(f"Không thể thêm ảnh vào dataset: {e_ds}")

                    if ref_path:
                        try:
                            from app.services.storytelling.face_extractor import extract_and_save_face_embedding
                            from app.services.storytelling.hardware_adapter import get_hardware_config
                            emb_path = os.path.join(ctx_mgr.chars_dir, _slug, "face.ipadpt.npy")
                            hw_config = get_hardware_config()
                            extract_and_save_face_embedding(ref_path, emb_path, device=hw_config["face_device"])
                            
                            char = ctx_mgr.get_character(_slug)
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

        # --- Dataset info panel cho nhân vật đã chọn ---
        if not is_new and selected_slug:
            _ds_count = ctx_mgr.count_dataset_images(selected_slug)
            _char_obj = ctx_mgr.get_character(selected_slug)
            if _char_obj:
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("📸 Ảnh Dataset", _ds_count)
                with col_info2:
                    _lora_badge = {"none": "⚪", "queued": "🟡", "training": "🔵",
                                   "trained": "🟢", "failed": "🔴"}.get(_char_obj.lora_status, "⚪")
                    st.metric("🧬 LoRA", f"{_lora_badge} {_char_obj.lora_status}")
                    if getattr(_char_obj, "ref_source", "manual") == "auto_bootstrap":
                        st.caption("🤖 Ảnh ref lấy TỰ ĐỘNG từ batch — upload ảnh mới ở form trên để thay nếu chưa ưng.")
                with col_info3:
                    _ac_val = st.checkbox("📥 Tự thu thập ảnh", value=_char_obj.auto_collect,
                                          key=f"ac_{selected_slug}")
                    if _ac_val != _char_obj.auto_collect:
                        _char_obj.auto_collect = _ac_val
                        ctx_mgr.save_context()
                if st.button("🗑 Xoá ảnh auto", key=f"del_auto_{selected_slug}"):
                    _ds_dir = ctx_mgr.get_dataset_dir(selected_slug)
                    _del_count = 0
                    for _f in os.listdir(_ds_dir):
                        if _f.startswith("auto_") and _f.endswith(".png"):
                            os.remove(os.path.join(_ds_dir, _f))
                            _del_count += 1
                    st.success(f"Đã xoá {_del_count} ảnh auto!")
                    st.rerun()

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
                                    clean_url = url.strip()
                                    if "](" in clean_url and clean_url.endswith(")"):
                                        clean_url = clean_url.split("](")[-1].rstrip(")")
                                    elif clean_url.startswith("<") and clean_url.endswith(">"):
                                        clean_url = clean_url[1:-1]
                                    try:
                                        st.image(clean_url, use_container_width=True)
                                    except Exception as e:
                                        st.warning("Không tải được ảnh")
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
                                            emb_path = os.path.join(ctx_mgr.chars_dir, slug, "face.ipadpt.npy")
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
        st.subheader("🧠 BỘ SINH ẢNH & MÔ HÌNH")
        
        # Image Generator Provider selection
        img_providers = ["Stable Diffusion (Local GPU)", "Local Gemini (API)"]
        current_img_provider = config.storytelling.get("image_gen_provider", "Stable Diffusion (Local GPU)")
        if current_img_provider not in img_providers:
            current_img_provider = "Stable Diffusion (Local GPU)"
            
        selected_img_provider = st.selectbox(
            "Bộ sinh ảnh (Image Generator Provider):",
            img_providers,
            index=img_providers.index(current_img_provider),
            help="Stable Diffusion sẽ chạy mô hình local trên GPU của bạn. Local Gemini (API) sẽ sử dụng Gemini API local (Imagen 3) không tốn VRAM GPU."
        )
        
        if selected_img_provider != current_img_provider:
            config.storytelling["image_gen_provider"] = selected_img_provider
            config.save_config()
            st.success(f"Đã cập nhật bộ sinh ảnh thành: {selected_img_provider}!")
            st.rerun()
            
        if selected_img_provider == "Stable Diffusion (Local GPU)":
            sd_model_options = {
                "Anything V5 (Anime chuyên dụng)": "stablediffusionapi/anything-v5",
                "DreamShaper 8 (Đa năng nhất — Anime + Semi-realistic)": "lykon/dreamshaper-8",
                "Realistic Vision 6 (Chân thực — Photorealistic)": "nyz3/Realistic_Vision_V6.0_B1_VAE",
                "CyberRealistic (Chân thực + Ánh sáng mạnh)": "cyberdelia/CyberRealistic",
            }
            sd_model_labels = list(sd_model_options.keys())
            sd_model_ids = list(sd_model_options.values())
            
            current_ckpt = ctx.checkpoint or "stablediffusionapi/anything-v5"
            current_idx = sd_model_ids.index(current_ckpt) if current_ckpt in sd_model_ids else 0
            
            selected_sd_label = st.selectbox(
                "Chọn mô hình Stable Diffusion:",
                sd_model_labels,
                index=current_idx,
                help="DreamShaper 8 được khuyến nghị cho đa phong cách. Anything V5 chỉ mạnh Anime. Realistic Vision cho ảnh chân thực."
            )
            new_ckpt = sd_model_options[selected_sd_label]
            
            if new_ckpt != current_ckpt:
                ctx.checkpoint = new_ckpt
                ctx_mgr.save_context(ctx)
                # Release old pipeline so the next warmup loads the new model
                try:
                    from app.services.storytelling.image_generator import StorytellingPipeline
                    instance = StorytellingPipeline._instance
                    if instance is not None:
                        instance.release()
                        StorytellingPipeline._instance = None
                        instance._initialized = False
                except Exception:
                    pass
                st.success(f"✅ Đã chuyển mô hình sang: **{selected_sd_label}**. Pipeline sẽ tự động tải mô hình mới khi sinh ảnh tiếp theo.")
                st.rerun()
            
            # Hiển thị thông tin model đang dùng
            model_short_name = selected_sd_label.split("(")[0].strip()
            model_strengths = {
                "Anything V5": "🎨 Anime 2D chuyên sâu — Nét vẽ sắc, màu rực. ⚠️ Thiên vị nữ, khó vẽ nam.",
                "DreamShaper 8": "⭐ Đa năng nhất — Anime, Fantasy, Semi-realistic, Watercolor. Vẽ nam tốt.",
                "Realistic Vision 6": "📷 Photorealistic — Ảnh như chụp thật. Vẽ người nam/nữ chân thực.",
                "CyberRealistic": "🔥 Chân thực cực cao — Ánh sáng phức tạp, kết cấu da tuyệt vời.",
            }
            st.caption(f"🔧 Model hiện tại: **{model_short_name}** — {model_strengths.get(model_short_name, '')}")
        else:
            # Local Gemini (API) active
            st.info("💡 Đang sử dụng **Local Gemini (API)**. Mô hình Imagen 3 (qua local API server) sẽ được gọi để sinh ảnh.")
            st.caption("⚠️ Lưu ý: Chế độ này không hỗ trợ khuôn mặt IP-Adapter. API server local cần đang chạy ở cổng được cấu hình (Global Settings).")
        
        st.markdown("---")
        st.subheader("📁 DỮ LIỆU ĐẦU VÀO & THỰC THI PIPELINE")
        
        from app.services.storytelling.orchestrator import StorytellingOrchestrator
        from app.services.storytelling.models import Scene, scene_from_dict
        orchestrator = StorytellingOrchestrator(ctx_mgr)
        current_state = orchestrator.load_state()
        
        exec_mode = st.radio(
            "⚡ Chế độ thực thi Pipeline:",
            [
                "🛑 3 Trạm Tương Tác (Human-in-the-Loop Studio)",
                "🚀 Sinh Video Tự Động (1 truyện hoặc cả thư mục)",
                "🖼️ Sinh 1 Ảnh (Single Image Generation)",
                "🖼️📦 Sinh Ảnh Hàng Loạt (Batch Image Generation)"
            ],
            index=0,
            horizontal=True,
            help="🚀 Sinh Video Tự Động = hợp nhất 'Chạy Tự Động Toàn Bộ' + 'Chạy Hàng Loạt' cũ: "
                 "tách cảnh bằng LLM, SRT tuỳ chọn (không có thì tự tạo từ kịch bản), "
                 "có resume + báo cáo + pre-flight LoRA."
        )
        
        # M3 (06/07/2026): Hai khối dưới ("⚡ Chạy Tự Động Toàn Bộ" và "📦 Chạy Hàng
        # Loạt" cũ) đã GỘP vào "🚀 Sinh Video Tự Động" — radio không còn 2 label này
        # nên code không bao giờ chạy; guard False để chắc chắn. Sẽ xoá hẳn sau khi
        # chế độ 🚀 được nghiệm thu ổn định.
        if False and exec_mode.startswith("⚡"):
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
        elif False and exec_mode.startswith("📦 Chạy"):  # M3: đã gộp vào 🚀
            st.markdown("### 📦 Chạy Hàng Loạt (Batch Processing)")
            st.info("Hệ thống sẽ tự động tìm các file .md và file audio (.mp3, .wav) trong thư mục Input, sắp xếp và ghép cặp chúng theo thứ tự tên file, sau đó render hàng loạt.")
            
            batch_input_dir = st.text_input("📁 Thư mục Input (chứa .md và audio)", placeholder="VD: D:\\Projects\\AudioBooks\\Chuong1_10")
            batch_output_dir = st.text_input("💾 Thư mục Output (nơi lưu video mp4)", placeholder="VD: D:\\Projects\\AudioBooks\\Output")
            fast_batch_mode = st.checkbox("⚡ Chế độ Siêu Tốc (Tắt upscale RealESRGAN, dùng bộ lọc Lanczos - Giữ SD trên VRAM để xử lý batch siêu nhanh)", value=True)
            
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
                                try:
                                    import gc, torch
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except: pass
                                
                                # Chạy pipeline
                                final_video = orchestrator.run_pipeline(temp_md, temp_audio, srt_path="", progress_callback=None, enable_upscaling=not fast_batch_mode)
                                
                                # Move và đổi tên sang output dir
                                out_file = os.path.join(batch_output_dir, f"{base_name}.mp4")
                                shutil.copy2(final_video, out_file)
                                success_count += 1
                                st.success(f"✅ Đã hoàn thành: {base_name}")
                                
                                # Dọn dẹp rác (nếu cần)
                                try: shutil.rmtree(temp_dir)
                                except: pass
                                try:
                                    import gc, torch
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
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
                edited_df = st.data_editor(df_scenes, width="stretch", num_rows="fixed", key="script_editor")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💾 Lưu Thay Đổi Kịch Bản", key="btn_save_script"):
                        for idx, row in edited_df.iterrows():
                            scenes_data[idx]["text_vi"] = row["Lời thoại (VI)"]
                            scenes_data[idx]["image_prompt"] = row["Prompt (EN)"]
                        scenes_obj = [scene_from_dict(s) for s in scenes_data]
                        orchestrator.save_state("SCRIPT_READY", scenes_obj, current_state["task_dir"], current_state.get("audio_path", ""), current_state.get("srt_path", ""), current_state.get("md_path", ""))
                        st.success("Đã lưu chỉnh sửa kịch bản!")
                with col_btn2:
                    if st.button("▶ Xác Nhận & Sang Trạm 2 (Sinh Ảnh SD)", key="btn_to_st2", type="primary"):
                        for idx, row in edited_df.iterrows():
                            scenes_data[idx]["text_vi"] = row["Lời thoại (VI)"]
                            scenes_data[idx]["image_prompt"] = row["Prompt (EN)"]
                        scenes_obj = [scene_from_dict(s) for s in scenes_data]
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
                            st.image(img_path, width="stretch")
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
                    scenes_obj = [scene_from_dict(s) for s in scenes_data]
                    orchestrator.save_state("RENDER_READY", scenes_obj, current_state["task_dir"], current_state.get("audio_path", ""), current_state.get("srt_path", ""), current_state.get("md_path", ""))
                    st.rerun()
                    
            elif step_status in ["RENDER_READY", "DONE"]:
                st.markdown("#### 🎬 Trạm 3 — Render Studio: Hậu Kỳ Upscale & Nhạc Nền")
                scenes_data = current_state.get("scenes", [])
                scenes_obj = [scene_from_dict(s) for s in scenes_data]
                
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

        elif exec_mode.startswith("🖼️ Sinh 1 Ảnh"):
            st.markdown("### 🖼️ Sinh 1 Ảnh (Single Image Generation)")
            
            from app.services.storytelling.image_gen_service import ImageGenOrchestrator
            from app.services.storytelling.models import ImageGenParams
            from app.services.storytelling.prompt_translator import PromptTranslator
            
            img_gen_orch = ImageGenOrchestrator(ctx_mgr)
            
            # --- TRẠM 1: Nhập Mô Tả ---
            st.markdown("#### 🚩 Trạm 1: Nhập Mô Tả & Cấu Hình")
            desc = st.text_area("📝 Mô tả hình ảnh", placeholder="VD: Dịch Phong đứng giữa phố đêm mưa, cầm ô...")
            st.caption("💡 Ghi tên nhân vật đã có trong Context Window để hệ thống tự động áp dụng khuôn mặt.")
            
            is_vietnamese = st.checkbox("☑ Mô tả bằng tiếng Việt (LLM sẽ dịch sang prompt EN chuẩn)", value=True)
            
            presets = PromptTranslator.list_available_presets()
            preset_options = ["Tự mô tả style (Không dùng preset)", "➕ Tạo Preset Mới bằng AI..."] + presets
            # Mặc định "storyboard" (flat pastel — sạch, đơn giản, hòa nền-nhân vật tốt)
            _default_idx = preset_options.index("storyboard") if "storyboard" in preset_options else (2 if len(preset_options) > 2 else 0)
            selected_preset = st.selectbox("🎨 Style Preset", preset_options, index=_default_idx)
            
            custom_style = ""
            if selected_preset == "Tự mô tả style (Không dùng preset)":
                custom_style = st.text_input("Tự mô tả style", placeholder="VD: Phong cách tranh sơn dầu cổ điển...")
            elif selected_preset == "➕ Tạo Preset Mới bằng AI...":
                st.info("AI sẽ tự động viết các tag chuẩn dựa trên mô tả của bạn và lưu thành 1 preset mới.")
                new_style_desc = st.text_input("Mô tả style mong muốn")
                new_preset_name = st.text_input("Tên preset (slug, viết liền không dấu, VD: cyberpunk_neon)")
                if st.button("🤖 Tạo Preset bằng AI"):
                    if new_style_desc and new_preset_name:
                        with st.spinner("Đang gọi AI..."):
                            success = img_gen_orch.translator.generate_custom_preset(new_style_desc, new_preset_name)
                            if success:
                                st.success(f"Đã tạo thành công preset: {new_preset_name}. Vui lòng reload lại trang!")
                            else:
                                st.error("Lỗi khi tạo preset.")
            
            with st.expander("⚙️ Tinh Chỉnh Nâng Cao"):
                st.caption("💡 Di chuột vào dấu (?) cạnh mỗi thông số để xem giải thích chi tiết. "
                           "Giá trị mặc định đọc từ config.toml — dùng chung với chế độ Batch.")
                # T2: config.toml là NGUỒN SỰ THẬT duy nhất cho tham số sinh ảnh
                _sg_cfg = load_storytelling_config()
                aspect_ratio = st.selectbox(
                    "Tỷ lệ ảnh (Aspect Ratio)",
                    ["16:9 (Landscape)", "9:16 (Portrait)", "1:1 (Square)", "4:3 (Ngang Cổ Điển)", "3:4 (Dọc Cổ Điển)", "21:9 (Cinematic)"],
                    index=0,
                    help="Khung hình của ảnh sinh ra (trước khi upscale). Lưu ý: SD1.5 được train ở ~512px nên khung càng rộng/dài (VD 21:9) càng dễ bị lặp nhân vật hoặc méo bố cục. 16:9 và 1:1 là an toàn nhất."
                )
                # SD1.5 train ở ~512px: cạnh dài <=768 để tránh lặp nhân vật/méo bố cục.
                # Độ phân giải cuối cùng do bước Upscale (Trạm 3) quyết định.
                ratio_map = {
                    "16:9 (Landscape)": (768, 432),
                    "9:16 (Portrait)": (432, 768),
                    "1:1 (Square)": (640, 640),
                    "4:3 (Ngang Cổ Điển)": (704, 528),
                    "3:4 (Dọc Cổ Điển)": (528, 704),
                    "21:9 (Cinematic)": (768, 328)
                }
                img_w, img_h = ratio_map[aspect_ratio]

                img_steps = st.slider(
                    "Inference Steps", 1, 50, int(_sg_cfg.get("num_inference_steps", 8)),
                    help="Số bước khử nhiễu — càng cao càng chi tiết nhưng càng chậm.\n\n• 1–14: tự động dùng chế độ NHANH (Hyper-SD LoRA) — hợp với 2/4/8 steps, nên để Guidance ≤ 1.5 (hoặc 5.0 với bản CFG-lora 8 steps).\n• 15–50: chế độ CHẤT LƯỢNG (DPM++ Karras) — 25 là điểm cân bằng tốt; trên 35 gần như không đẹp thêm."
                )
                img_guidance = st.slider(
                    "Guidance Scale", 0.0, 20.0, float(_sg_cfg.get("guidance_scale", 5.0)), 0.5,
                    help="Mức độ 'nghe lời' prompt (CFG).\n\n• ≤ 1.0: TẮT CFG — nhanh gấp đôi nhưng Negative Prompt bị BỎ QUA hoàn toàn (chỉ dùng với chế độ nhanh Hyper-SD).\n• 5–8: chuẩn cho chế độ chất lượng — bám prompt tốt, màu tự nhiên.\n• > 10: bám prompt quá gắt → màu cháy, tương phản gắt, ảnh 'nhựa'."
                )
                ip_scale = st.slider(
                    "Mức độ giống nhân vật (IP-Adapter)", 0.0, 1.0, float(_sg_cfg.get("ip_adapter_scale", 0.6)), 0.05,
                    help="Trọng số của ẢNH THAM CHIẾU nhân vật so với prompt.\n\n• 0: bỏ qua ảnh tham chiếu (chỉ dùng prompt).\n• 0.5–0.7: khuyên dùng — giữ nét mặt/kiểu tóc nhưng vẫn cho phép đổi pose, biểu cảm, bối cảnh.\n• > 0.8: rất giống ảnh gốc nhưng 'dính' luôn cả tư thế, góc mặt, trang phục của ảnh tham chiếu — bối cảnh và hành động trong prompt bị lấn át."
                )
                img_seed = st.number_input(
                    "Seed (-1 = Random)", value=-1,
                    help="Hạt giống ngẫu nhiên. Cùng seed + cùng prompt + cùng thông số = ra ĐÚNG ảnh đó lại (dùng để tái tạo ảnh ưng ý). Đặt -1 để mỗi lần sinh ra một biến thể mới."
                )
                sg_face_detail = st.checkbox(
                    "🎭 Vẽ lại mặt tự động (Face Detailer)", value=bool(_sg_cfg.get("enable_face_detailer", True)), key="sg_fd",
                    help="Tự phát hiện khuôn mặt trong ảnh → phóng lên 512px → vẽ lại bằng chính model + IP-Adapter identity → dán về vị trí cũ. Cứu mặt nhỏ/méo trong cảnh wide-shot. Tốn thêm ~2-4s mỗi mặt. Trong batch luôn chạy tự động theo cấu hình này."
                )
                if st.button("💾 Lưu làm mặc định (dùng chung cho Batch)", key="sg_save_cfg",
                             help="Ghi Steps / Guidance / IP-Adapter / Face Detailer hiện tại vào config.toml — batch video và các phiên sau dùng đúng bộ tham số này."):
                    config.storytelling["num_inference_steps"] = int(img_steps)
                    config.storytelling["guidance_scale"] = float(img_guidance)
                    config.storytelling["ip_adapter_scale"] = float(ip_scale)
                    config.storytelling["enable_face_detailer"] = bool(sg_face_detail)
                    config.save_config()
                    st.success(f"✅ Đã lưu: steps={img_steps}, guidance={img_guidance}, ip={ip_scale}, detailer={'bật' if sg_face_detail else 'tắt'}")

            gen_params = ImageGenParams(
                width=img_w, height=img_h, num_inference_steps=img_steps,
                guidance_scale=img_guidance, seed=img_seed, ip_adapter_scale=ip_scale,
                enable_face_detailer=sg_face_detail
            )
            
            if st.button("▶ Xử Lý Prompt & Sang Trạm 2"):
                if desc:
                    with st.spinner("Đang xử lý..."):
                        final_preset = "" if selected_preset.startswith("Tự mô tả") or selected_preset.startswith("➕") else selected_preset
                        tasks = img_gen_orch.step1_prepare_prompts([desc], final_preset, custom_style, is_vietnamese)
                        st.session_state["single_img_tasks"] = tasks
                else:
                    st.warning("Vui lòng nhập mô tả!")
                    
            # --- TRẠM 2: Review & Sinh Ảnh ---
            if "single_img_tasks" in st.session_state and st.session_state["single_img_tasks"]:
                st.markdown("---")
                st.markdown("#### 🚩 Trạm 2: Review Prompt & Sinh Ảnh")
                task = st.session_state["single_img_tasks"][0]
                
                st.text_area("✅ Prompt EN", value=task.processed_prompt, height=100, disabled=True)
                st.text_area("❌ Negative Prompt", value=task.negative_prompt, height=50, disabled=True)
                
                if task.character_slugs:
                    st.success(f"👤 Nhân vật matched: {', '.join(task.character_slugs)} (Primary: {task.primary_character})")
                else:
                    st.info("👤 Không tìm thấy nhân vật nào. Sẽ tạo ảnh ngẫu nhiên.")
                    
                col_gen, col_reroll = st.columns(2)
                with col_gen:
                    if st.button("▶ Sinh Ảnh", type="primary"):
                        with st.spinner("Đang sinh ảnh..."):
                            results = img_gen_orch.step2_generate_images([task], gen_params)
                            for _w in img_gen_orch.get_pipeline_warnings():
                                st.warning(_w)
                            if results:
                                st.session_state["single_img_result"] = results[0]
                            else:
                                st.error("Sinh ảnh thất bại. Xem log terminal để biết chi tiết.")
                with col_reroll:
                    if st.button("🔄 Re-roll (Seed mới)"):
                        if "single_img_result" in st.session_state:
                            with st.spinner("Đang sinh lại..."):
                                gen_params.seed = -1
                                res = img_gen_orch.reroll_image(task, gen_params)
                                if res:
                                    st.session_state["single_img_result"] = res
                                    
                if "single_img_result" in st.session_state:
                    res = st.session_state["single_img_result"]
                    st.image(res.image_path, caption=f"Seed: {res.seed}")
                    
                    if st.button("✅ Chấp nhận & Sang Trạm 3"):
                        # T6: thu thập dataset chuyển sang SAU upscale
                        # (bên trong step3_upscale_export) — không lấy ảnh draft nữa.
                        st.session_state["single_img_ready"] = True
                        
            # --- TRẠM 3: Upscale ---
            if st.session_state.get("single_img_ready"):
                st.markdown("---")
                st.markdown("#### 🚩 Trạm 3: Upscale & Export")
                
                en_upscale = st.checkbox("☑ Bật AI Upscaler (RealESRGAN 4x)", value=True, key="sg_up")
                target_q = st.radio("📐 Chất lượng đầu ra", ["1k", "2k", "4k"], index=1, horizontal=True)
                
                output_dir = st.session_state.get("output_folder", "storage/generated_images")
                st.info(f"📁 Output: {output_dir}")
                
                if st.button("🎬 Export Ảnh Final", type="primary"):
                    with st.spinner("Đang Upscale và Export..."):
                        final_paths = img_gen_orch.step3_upscale_export(
                            [st.session_state["single_img_result"]], 
                            enable_upscale=en_upscale, target_quality=target_q, output_dir=output_dir
                        )
                        if final_paths:
                            st.success(f"✅ Đã lưu: {final_paths[0]}")
                            st.image(final_paths[0])
                            
        elif exec_mode.startswith("🖼️📦 Sinh Ảnh Hàng Loạt"):
            st.markdown("### 🖼️📦 Sinh Ảnh Hàng Loạt (Batch Image Generation)")
            from app.services.storytelling.image_gen_service import ImageGenOrchestrator
            from app.services.storytelling.models import ImageGenParams
            from app.services.storytelling.prompt_translator import PromptTranslator
            img_gen_orch = ImageGenOrchestrator(ctx_mgr)
            
            # --- TRẠM 1: Nhập Mô Tả ---
            st.markdown("#### 🚩 Trạm 1: Nhập Danh Sách Mô Tả")
            batch_desc = st.text_area("📝 Mỗi mô tả cách nhau bằng 1 HÀNG TRỐNG", height=300, 
                                    placeholder="Dịch Phong uống trà...\n\nLạc Lan Tuyết cầm kiếm...")
            is_vietnamese = st.checkbox("☑ Mô tả bằng tiếng Việt", value=True, key="bat_vi")
            
            presets = PromptTranslator.list_available_presets()
            _preset_opts = ["Tự mô tả style"] + presets
            _default_idx = _preset_opts.index("storyboard") if "storyboard" in _preset_opts else (1 if len(presets) > 0 else 0)
            selected_preset = st.selectbox("🎨 Style Preset (áp dụng cho tất cả)", _preset_opts, index=_default_idx)
            custom_style = ""
            if selected_preset == "Tự mô tả style":
                custom_style = st.text_input("Tự mô tả style", key="bat_cs")
                
            with st.expander("⚙️ Tinh Chỉnh Chung"):
                aspect_ratio = st.selectbox(
                    "Tỷ lệ ảnh (Aspect Ratio)",
                    ["16:9 (Landscape)", "9:16 (Portrait)", "1:1 (Square)", "4:3 (Ngang Cổ Điển)", "3:4 (Dọc Cổ Điển)", "21:9 (Cinematic)"],
                    index=0, key="b_aspect"
                )
                # SD1.5 train ở ~512px: cạnh dài <=768 để tránh lặp nhân vật/méo bố cục.
                # Độ phân giải cuối cùng do bước Upscale (Trạm 3) quyết định.
                ratio_map = {
                    "16:9 (Landscape)": (768, 432),
                    "9:16 (Portrait)": (432, 768),
                    "1:1 (Square)": (640, 640),
                    "4:3 (Ngang Cổ Điển)": (704, 528),
                    "3:4 (Dọc Cổ Điển)": (528, 704),
                    "21:9 (Cinematic)": (768, 328)
                }
                img_w, img_h = ratio_map[aspect_ratio]
                
                # T2: default đọc từ config.toml — cùng nguồn sự thật với chế độ sinh đơn & batch video
                _bat_cfg = load_storytelling_config()
                img_steps = st.slider(
                    "Inference Steps", 1, 50, int(_bat_cfg.get("num_inference_steps", 8)), key="bs",
                    help="Số bước khử nhiễu — càng cao càng chi tiết nhưng càng chậm.\n\n• 1–14: chế độ NHANH (Hyper-SD LoRA) — nên để Guidance ≤ 1.5 (hoặc 5.0 với bản CFG-lora 8 steps).\n• 15–50: chế độ CHẤT LƯỢNG (DPM++ Karras) — 25 là điểm cân bằng; với batch lớn, giảm steps là cách tiết kiệm thời gian rõ nhất."
                )
                img_guidance = st.slider(
                    "Guidance Scale", 0.0, 20.0, float(_bat_cfg.get("guidance_scale", 5.0)), 0.5, key="bg",
                    help="Mức độ 'nghe lời' prompt (CFG).\n\n• ≤ 1.0: TẮT CFG — nhanh gấp đôi nhưng Negative Prompt bị BỎ QUA (chỉ dùng với chế độ nhanh).\n• 5–8: chuẩn cho chế độ chất lượng.\n• > 10: màu cháy, tương phản gắt."
                )
                ip_scale = st.slider(
                    "Mức độ giống nhân vật (IP-Adapter)", 0.0, 1.0, float(_bat_cfg.get("ip_adapter_scale", 0.6)), 0.05, key="bip",
                    help="Trọng số của ẢNH THAM CHIẾU nhân vật so với prompt (áp dụng cho mọi ảnh trong batch).\n\n• 0.5–0.7: khuyên dùng — giữ nét mặt nhưng vẫn đổi được pose/bối cảnh theo từng mô tả.\n• > 0.8: các ảnh trong batch sẽ na ná ảnh tham chiếu, mất đa dạng cảnh."
                )

                bat_face_detail = st.checkbox(
                    "🎭 Vẽ lại mặt tự động (Face Detailer)", value=bool(_bat_cfg.get("enable_face_detailer", True)), key="bat_fd",
                    help="Tự phát hiện mặt → vẽ lại 512px với identity → dán về. Cứu mặt nhỏ/méo trong wide-shot. Thêm ~2-4s mỗi mặt, chạy hoàn toàn tự động cho cả batch."
                )
                if st.button("💾 Lưu làm mặc định (dùng chung mọi chế độ)", key="bat_save_cfg"):
                    config.storytelling["num_inference_steps"] = int(img_steps)
                    config.storytelling["guidance_scale"] = float(img_guidance)
                    config.storytelling["ip_adapter_scale"] = float(ip_scale)
                    config.storytelling["enable_face_detailer"] = bool(bat_face_detail)
                    config.save_config()
                    st.success(f"✅ Đã lưu: steps={img_steps}, guidance={img_guidance}, ip={ip_scale}, detailer={'bật' if bat_face_detail else 'tắt'}")

            gen_params = ImageGenParams(width=img_w, height=img_h, num_inference_steps=img_steps, guidance_scale=img_guidance, seed=-1, ip_adapter_scale=ip_scale, enable_face_detailer=bat_face_detail)

            batch_folder = st.text_input("📁 Tên thư mục output (bên trong Global Output)", value="batch_images_01")
            
            if st.button("▶ Xử Lý Tất Cả → Sang Trạm 2"):
                raw_list = [d.strip() for d in batch_desc.split("\n\n") if d.strip()]
                if raw_list:
                    if len(raw_list) > 50:
                        st.warning(f"Lượng mô tả lớn ({len(raw_list)}). Có thể mất nhiều thời gian!")
                    with st.spinner(f"Đang xử lý {len(raw_list)} mô tả..."):
                        final_preset = "" if selected_preset == "Tự mô tả style" else selected_preset
                        tasks = img_gen_orch.step1_prepare_prompts(raw_list, final_preset, custom_style, is_vietnamese)
                        st.session_state["batch_img_tasks"] = tasks
                else:
                    st.warning("Vui lòng nhập mô tả!")
                    
            # --- TRẠM 2: Sinh Ảnh Hàng Loạt ---
            if "batch_img_tasks" in st.session_state and st.session_state["batch_img_tasks"]:
                tasks = st.session_state["batch_img_tasks"]
                st.markdown("---")
                st.markdown(f"#### 🚩 Trạm 2: Sinh {len(tasks)} Ảnh")
                
                # Preview bảng
                st.write("📋 **Preview danh sách task:**")
                preview_data = [{"ID": t.task_id, "Gốc": t.original_description[:50]+"...", "Prompt": t.processed_prompt[:50]+"...", "Nhân vật": ",".join(t.character_slugs)} for t in tasks]
                st.table(preview_data)
                
                if st.button("▶ Sinh Tất Cả Ảnh", type="primary"):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    def update_ui(curr, tot):
                        progress_text.text(f"Đang sinh ảnh {curr}/{tot}...")
                        progress_bar.progress(curr / max(tot, 1))
                        
                    results = img_gen_orch.step2_generate_images(tasks, gen_params, progress_callback=update_ui)
                    for _w in img_gen_orch.get_pipeline_warnings():
                        st.warning(_w)
                    if len(results) < len(tasks):
                        st.error(f"⚠️ Chỉ sinh thành công {len(results)}/{len(tasks)} ảnh. Xem log terminal để biết chi tiết.")
                    st.session_state["batch_img_results"] = results
                    
                if "batch_img_results" in st.session_state:
                    st.success("✅ Đã sinh xong toàn bộ ảnh!")
                    # Show grid
                    results = st.session_state["batch_img_results"]
                    cols = st.columns(3)
                    for idx, res in enumerate(results):
                        with cols[idx % 3]:
                            st.image(res.image_path, caption=f"ID: {res.task_id} | Seed: {res.seed}")
                            
                    if st.button("✅ Chấp nhận tất cả & Sang Trạm 3"):
                        # T6: thu thập dataset chuyển sang SAU upscale
                        # (bên trong step3_upscale_export) — không lấy ảnh draft nữa.
                        st.session_state["batch_img_ready"] = True
                        
            # --- TRẠM 3: Upscale ---
            if st.session_state.get("batch_img_ready") and "batch_img_results" in st.session_state:
                st.markdown("---")
                st.markdown("#### 🚩 Trạm 3: Upscale & Export Hàng Loạt")
                
                en_upscale = st.checkbox("☑ Bật AI Upscaler", value=True, key="b_up")
                target_q = st.radio("📐 Chất lượng đầu ra", ["1k", "2k", "4k"], index=1, horizontal=True, key="bq")
                
                output_dir = st.session_state.get("output_folder", "storage/generated_images")
                
                if st.button("🎬 Export Tất Cả", type="primary"):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    def update_ui2(curr, tot):
                        progress_text.text(f"Đang upscale và export {curr}/{tot}...")
                        progress_bar.progress(curr / max(tot, 1))
                        
                    final_paths = img_gen_orch.step3_upscale_export(
                        st.session_state["batch_img_results"], 
                        enable_upscale=en_upscale, target_quality=target_q, 
                        output_dir=output_dir, batch_folder_name=batch_folder,
                        progress_callback=update_ui2
                    )
                    st.success(f"✅ Đã export {len(final_paths)} ảnh vào {os.path.join(output_dir, batch_folder)}")

        elif exec_mode.startswith("🚀"):
            st.markdown("### 🚀 Sinh Video Tự Động")
            st.caption("Tách cảnh bằng LLM theo ngữ cảnh (8-15s/cảnh) • SRT tuỳ chọn — không có sẽ TỰ TẠO phụ đề từ kịch bản • Resume khi chạy lại • Báo cáo cuối batch.")

            from app.services.storytelling.batch_video_runner import BatchItem, scan_batch_dir, run_batch

            output_dir = st.text_input(
                "💾 Thư mục output (video .mp4 + báo cáo)",
                value="F:\\Output\\batch_videos",
                key="bv_output_dir"
            )

            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                bv_upscale = st.checkbox("🔍 Upscale ảnh (RealESRGAN)", value=True, key="bv_up",
                                         help="Tắt = chế độ siêu tốc: xuất thẳng ảnh draft, nhanh hơn đáng kể nhưng ảnh mềm hơn.")
            with col_opt2:
                bv_burn_sub = st.checkbox("💬 Gắn phụ đề vào video", value=True, key="bv_sub")
            with col_opt3:
                bv_fresh = st.checkbox("🔄 Làm mới (bỏ resume)", value=False, key="bv_fresh",
                                       help="Xoá trạng thái batch cũ, tách cảnh + sinh lại TỪ ĐẦU. Dùng khi đổi kịch bản/tham số hoặc lần chạy trước ra kết quả sai. Không tick = tiếp tục từ chỗ dở.")

            bv_items = []
            tab_single, tab_folder = st.tabs(["📄 1 Truyện", "📁 Cả Thư Mục"])

            with tab_single:
                col_md, col_audio, col_srt = st.columns(3)
                with col_md:
                    sv_md = st.file_uploader("Kịch bản (.md)", type=["md"], key="sv_md")
                with col_audio:
                    sv_audio = st.file_uploader("Audio (.mp3/.wav)", type=["mp3", "wav"], key="sv_audio")
                with col_srt:
                    sv_srt = st.file_uploader("Phụ đề (.srt) — tuỳ chọn", type=["srt"], key="sv_srt")
                if sv_md and sv_audio:
                    _sv_temp = os.path.join(ctx_mgr.context_dir, "temp", "single_auto")
                    os.makedirs(_sv_temp, exist_ok=True)
                    _sv_md_path = os.path.join(_sv_temp, sv_md.name)
                    _sv_audio_path = os.path.join(_sv_temp, sv_audio.name)
                    with open(_sv_md_path, "wb") as f: f.write(sv_md.getbuffer())
                    with open(_sv_audio_path, "wb") as f: f.write(sv_audio.getbuffer())
                    _sv_srt_path = ""
                    if sv_srt:
                        _sv_srt_path = os.path.join(_sv_temp, sv_srt.name)
                        with open(_sv_srt_path, "wb") as f: f.write(sv_srt.getbuffer())
                    bv_items = [BatchItem(
                        stem=os.path.splitext(sv_md.name)[0],
                        md_path=_sv_md_path, audio_path=_sv_audio_path, srt_path=_sv_srt_path,
                    )]
                    st.success(f"Sẵn sàng: {bv_items[0].stem} (SRT: {'có' if _sv_srt_path else 'tự tạo từ kịch bản'})")

            with tab_folder:
                batch_dir = st.text_input(
                    "📁 Thư mục chứa các cặp file trùng tên (ep01.md + ep01.wav [+ ep01.srt])",
                    value="F:\\Du_An_Truyen_Chu\\Nguoi_Tren_Van_Nguoi\\Audio",
                    key="bv_batch_dir"
                )
                if batch_dir and os.path.isdir(batch_dir):
                    _folder_items = scan_batch_dir(batch_dir)
                    if _folder_items:
                        st.markdown(f"**Tìm thấy {len(_folder_items)} item:**")
                        st.table([{
                            "Stem": it.stem,
                            "MD": os.path.basename(it.md_path),
                            "Audio": os.path.basename(it.audio_path),
                            "SRT": os.path.basename(it.srt_path) if it.srt_path else "tự tạo"
                        } for it in _folder_items])
                        bv_items = _folder_items
                    else:
                        st.warning("⚠️ Không tìm thấy cặp md+audio nào trong thư mục.")
                elif batch_dir:
                    st.error("Thư mục không tồn tại!")

            if bv_items:
                if st.button(f"▶ Chạy ({len(bv_items)} video)", type="primary", key="bv_run"):
                    if bv_fresh:
                        _bv_state_file = os.path.join(output_dir, "batch_state.json")
                        if os.path.exists(_bv_state_file):
                            try:
                                os.remove(_bv_state_file)
                                st.info("🔄 Đã xoá trạng thái batch cũ — chạy lại từ đầu.")
                            except Exception as _e_st:
                                st.warning(f"Không xoá được batch_state.json: {_e_st}")
                    bv_progress = st.empty()
                    bv_bar = st.progress(0)

                    def bv_progress_cb(msg, pct):
                        bv_progress.text(msg)
                        bv_bar.progress(min(max(pct, 0) / 100.0, 1.0))

                    st.warning("⚠️ Đang chạy. KHÔNG đóng tab hoặc thao tác UI. "
                               "Nếu bị ngắt giữa chừng, chạy lại sẽ tiếp tục từ chỗ dở (resume).")

                    bv_report = run_batch(
                        ctx_mgr=ctx_mgr,
                        items=bv_items,
                        output_dir=output_dir,
                        options={
                            "use_semantic_split": True,   # chính sách 06/07: LLM luôn bật
                            "use_whisper": False,          # không Whisper — SRT tự tạo từ kịch bản
                            "enable_upscale": bv_upscale,
                            "burn_subtitles": bv_burn_sub,
                        },
                        progress_cb=bv_progress_cb,
                    )

                    st.markdown("### 📊 Báo cáo")
                    for rpt in bv_report.items:
                        status_icon = "✅" if rpt["status"] == "video_done" else "❌"
                        st.write(
                            f"{status_icon} **{rpt['stem']}** — "
                            f"{rpt['status']} ({rpt.get('time_seconds', 0):.0f}s)"
                        )
                        if rpt.get("error"):
                            st.error(f"  Lỗi: {rpt['error']}")
                    st.success(f"Đã xong! Output: {output_dir}")


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
    
    enable_audio_cleanup_t = st.checkbox("Bật làm sạch âm thanh (Tách tạp âm bằng Demucs)", value=False, key="t_audio_clean", help="Sử dụng AI Demucs để tách giọng nói khỏi nhạc nền/hiệu ứng âm thanh, giúp Whisper nhận diện chính xác hơn.")
    
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
                    logger.info(f"Parameters: enable_voiceover={enable_voiceover_t}, tts_engine={tts_engine_t}, tts_voice={tts_voice_t}, ducking_ratio={ducking_ratio_t}, auto_clone={auto_clone_t}, clean_audio={enable_audio_cleanup_t}")
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
                        auto_clone=auto_clone_t,
                        clean_audio=enable_audio_cleanup_t
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

with tab6:
    st.header("Workflow 6: Video Upscale (Trạm 3 & Demucs)")
    st.markdown("Tính năng này giúp làm nét video hoạt hình bằng **Workflow 5 Trạm 3**, đồng thời có khả năng sử dụng **Demucs** để giữ lại giọng nói và loại bỏ tạp âm/nhạc nền. Cuối cùng, ghép nối tự động bằng H264 NVENC GPU.")
    
    upscale_input_video = st.text_input("Đường dẫn Video đầu vào (.mp4, .mkv)", placeholder="Ví dụ: D:/video/hoathinh.mp4", key="upscale_input_video")
    upscale_output_dir = st.text_input("Thư mục lưu kết quả", value=st.session_state.get("output_folder", ""), key="upscale_output_dir")
    upscale_output_name = st.text_input("Tên Video đầu ra (Tùy chọn, để trống sẽ tự lấy tên gốc + _upscaled)", key="upscale_output_name")
    
    upscale_resolution = st.selectbox(
        "Độ phân giải đầu ra (Scale bằng GPU NVENC)",
        ["Gốc", "720p (HD)", "1080p (Full HD)", "1440p (2K)", "2160p (4K)"],
        index=0,
        key="upscale_resolution"
    )
    
    upscale_method = st.selectbox(
        "Phương pháp Làm nét", 
        [
            "Chỉ phóng to (Nhanh nhất)", 
            "Làm nét nhanh (FFmpeg CAS)", 
            "AI Siêu nhẹ (AnimeVideo-V3 - Mượt & Nhanh)", 
            "AI Cao cấp (Real-ESRGAN - Nét căng & Rất chậm)"
        ],
        index=2,
        key="upscale_method"
    )
    use_realesrgan = ("AI" in upscale_method)
    
    # Lựa chọn Tile Size cho VRAM
    tile_size = 512
    if use_realesrgan:
        tile_options = {
            "Siêu thấp (Tile 256) - Phù hợp VRAM 2-4GB": 256,
            "Trung bình (Tile 512) - Phù hợp VRAM 4-6GB (An toàn nhất)": 512,
            "Cao (Tile 768) - Phù hợp VRAM 8GB": 768,
            "Tối đa (Tile 0 - Tắt Tiling) - Phù hợp VRAM >12GB": 0
        }
        selected_tile = st.selectbox(
            "Tối ưu hóa bộ nhớ VRAM cho Card đồ họa của bạn", 
            list(tile_options.keys()), 
            index=1, 
            key="upscale_tile_size"
        )
        tile_size = tile_options[selected_tile]
        
    use_demucs = st.checkbox("🎧 Sử dụng Demucs để làm sạch tạp âm/nhạc nền (Giữ lại Vocal)", value=False, key="upscale_use_demucs")
    
    if st.button("🚀 Bắt đầu làm nét Video", key="btn_start_upscale", type="primary"):
        if not upscale_input_video or not os.path.exists(upscale_input_video):
            st.error("Vui lòng nhập đường dẫn video hợp lệ!")
        else:
            task_id_upscale = str(uuid.uuid4())
            task_dir_upscale = utils.task_dir(task_id_upscale)
            os.makedirs(task_dir_upscale, exist_ok=True)
            log_path_upscale = os.path.join(task_dir_upscale, "upscale_log.txt")
            
            st.info(f"Đang xử lý tại thư mục tạm: {task_dir_upscale}")
            log_placeholder_upscale = st.empty()
            
            # Khởi tạo kết quả
            result_upscale = {"done": False, "error": None, "video": None}
            
            def run_upscale_in_thread():
                sink_id = None
                try:
                    sink_id = logger.add(log_path_upscale, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
                    logger.info("=== Bắt đầu Workflow 6 (Video Upscale) ===")
                    logger.info(f"Video đầu vào: {upscale_input_video}")
                    logger.info(f"Sử dụng Demucs: {'Có' if use_demucs else 'Không'}")
                    
                    # Add video_processor to sys.path implicitly by executing it via import
                    import sys
                    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                    if root_dir not in sys.path:
                        sys.path.insert(0, root_dir)
                    from src.utils.video_processor import process_animation_video
                    
                    input_base_name = os.path.splitext(os.path.basename(upscale_input_video))[0]
                    final_name = (upscale_output_name or f"{input_base_name}_upscaled").strip() + ".mp4"
                    os.makedirs(upscale_output_dir, exist_ok=True)
                    out_path = os.path.join(upscale_output_dir, final_name)
                    
                    success = process_animation_video(
                        input_path=upscale_input_video,
                        output_path=out_path,
                        temp_dir=task_dir_upscale,
                        use_demucs=use_demucs,
                        upscale_method=upscale_method,
                        tile_size=tile_size,
                        resolution=upscale_resolution,
                        log_callback=logger.info
                    )
                    
                    if success:
                        result_upscale["video"] = out_path
                        logger.info("=== Hoàn thành Workflow 6 thành công! ===")
                    else:
                        raise Exception("Lỗi xử lý làm nét video.")
                        
                except Exception as ex:
                    import traceback
                    logger.error(f"Lỗi: {ex}")
                    logger.error(traceback.format_exc())
                    result_upscale["error"] = ex
                finally:
                    result_upscale["done"] = True
                    if sink_id:
                        try:
                            logger.remove(sink_id)
                        except Exception:
                            pass
            
            # Start thread
            thread_upscale = threading.Thread(target=run_upscale_in_thread, name=f"task_{task_id_upscale}")
            thread_upscale.start()
            
            # Poll logs
            while not result_upscale["done"]:
                time.sleep(0.5)
                if os.path.exists(log_path_upscale):
                    with open(log_path_upscale, "r", encoding="utf-8", errors="ignore") as f:
                        log_placeholder_upscale.text(f.read())
            
            # Final poll
            if os.path.exists(log_path_upscale):
                with open(log_path_upscale, "r", encoding="utf-8", errors="ignore") as f:
                    log_placeholder_upscale.text(f.read())
                    
            if result_upscale["error"]:
                st.error(f"Lỗi làm nét video: {result_upscale['error']}")
            elif result_upscale["video"]:
                final_video_path = result_upscale["video"]
                st.success("🎉 Làm nét video thành công!")
                st.info(f"📁 Video kết quả đã được lưu tại: `{final_video_path}`")
                st.video(final_video_path)
                
                with open(final_video_path, "rb") as f:
                    st.download_button(
                        label="📥 Tải xuống Video",
                        data=f,
                        file_name=os.path.basename(final_video_path),
                        mime="video/mp4",
                        key="dl_upscaled_video_immediate"
                    )

with tab7:
    st.header("Workflow 7: Tạo Ảnh AI (Text-to-Image)")
    st.markdown("Nhập prompt, chọn mô hình và tinh chỉnh thông số sinh ảnh — không cần tạo bộ truyện hay nhân vật.")

    qi_provider = config.storytelling.get("image_gen_provider", "Stable Diffusion (Local GPU)")
    if qi_provider == "Local Gemini (API)":
        st.warning("⚠️ Bộ sinh ảnh toàn cục đang là **Local Gemini (API)** (đổi trong tab AI Storytelling → BỘ SINH ẢNH & MÔ HÌNH). "
                   "Ảnh sẽ được sinh qua Gemini API — mô hình SD và các thông số bên dưới sẽ bị BỎ QUA.")

    # --- Chọn mô hình ---
    qi_model_options = {
        "Anything V5 (Anime chuyên dụng)": "stablediffusionapi/anything-v5",
        "DreamShaper 8 (Đa năng nhất — Anime + Semi-realistic)": "lykon/dreamshaper-8",
        "Realistic Vision 6 (Chân thực — Photorealistic)": "nyz3/Realistic_Vision_V6.0_B1_VAE",
        "CyberRealistic (Chân thực + Ánh sáng mạnh)": "cyberdelia/CyberRealistic",
        "Tùy chỉnh (Repo HuggingFace / đường dẫn local)...": "__custom__",
    }
    qi_model_label = st.selectbox(
        "🧠 Mô hình sinh ảnh (Checkpoint)",
        list(qi_model_options.keys()),
        index=1,
        key="qi_model",
        help="Model chỉ được tải/đổi khi bấm Tạo Ảnh. Lần đầu dùng một model sẽ mất vài phút để tải về."
    )
    qi_checkpoint = qi_model_options[qi_model_label]
    if qi_checkpoint == "__custom__":
        qi_checkpoint = st.text_input(
            "Repo HuggingFace hoặc đường dẫn model cục bộ",
            placeholder="VD: lykon/dreamshaper-8 hoặc D:/models/my_model",
            key="qi_custom_ckpt"
        ).strip()

    # --- Prompt ---
    qi_prompt = st.text_area(
        "📝 Prompt (tiếng Anh — hỗ trợ cú pháp trọng số (tag:1.3))",
        height=120,
        placeholder="VD: 1girl, silver hair, hanfu, ancient chinese palace, moonlight, cinematic lighting",
        key="qi_prompt"
    )
    qi_neg = st.text_area(
        "❌ Negative Prompt",
        value="(worst quality, low quality:1.4), lowres, bad anatomy, bad hands, extra fingers, watermark, text, jpeg artifacts",
        height=70,
        key="qi_neg",
        help="Chỉ có tác dụng khi Guidance Scale > 1.0 (CFG bật)."
    )

    # --- Thông số sinh ảnh ---
    with st.expander("⚙️ Thông số sinh ảnh", expanded=True):
        _qi_cfg = load_storytelling_config()
        col_size, col_num = st.columns(2)
        with col_size:
            qi_aspect = st.selectbox(
                "Tỷ lệ ảnh (Aspect Ratio)",
                ["16:9 (Landscape)", "9:16 (Portrait)", "1:1 (Square)", "4:3 (Ngang Cổ Điển)", "3:4 (Dọc Cổ Điển)", "21:9 (Cinematic)", "Kích thước tùy chỉnh..."],
                index=0,
                key="qi_aspect",
                help="SD1.5 được train ở ~512px nên khung càng rộng/dài càng dễ lặp nhân vật hoặc méo bố cục. 16:9 và 1:1 an toàn nhất."
            )
        with col_num:
            qi_num = st.slider("Số ảnh muốn tạo", 1, 8, 1, key="qi_num",
                               help="Sinh tuần tự từng ảnh để tiết kiệm VRAM. Seed cố định sẽ tự +1 cho mỗi ảnh tiếp theo.")

        # Cạnh dài <=768 cho SD1.5 (giống 2 chế độ sinh ảnh của Storytelling)
        qi_ratio_map = {
            "16:9 (Landscape)": (768, 432),
            "9:16 (Portrait)": (432, 768),
            "1:1 (Square)": (640, 640),
            "4:3 (Ngang Cổ Điển)": (704, 528),
            "3:4 (Dọc Cổ Điển)": (528, 704),
            "21:9 (Cinematic)": (768, 328),
        }
        if qi_aspect == "Kích thước tùy chỉnh...":
            col_w, col_h = st.columns(2)
            with col_w:
                qi_w = st.number_input("Chiều rộng (px)", min_value=256, max_value=1024, value=768, step=8, key="qi_w")
            with col_h:
                qi_h = st.number_input("Chiều cao (px)", min_value=256, max_value=1024, value=432, step=8, key="qi_h")
        else:
            qi_w, qi_h = qi_ratio_map[qi_aspect]
            st.caption(f"📐 Kích thước sinh: **{qi_w} × {qi_h}px**")

        qi_steps = st.slider(
            "Inference Steps", 1, 50, int(_qi_cfg.get("num_inference_steps", 8)), key="qi_steps",
            help="Số bước khử nhiễu — càng cao càng chi tiết nhưng càng chậm.\n\n• 1–14: tự động dùng chế độ NHANH (Hyper-SD LoRA) — hợp với 2/4/8 steps, nên để Guidance ≤ 1.5 (hoặc 5.0 với bản CFG-lora 8 steps).\n• 15–50: chế độ CHẤT LƯỢNG (DPM++ Karras) — 25 là điểm cân bằng tốt; trên 35 gần như không đẹp thêm."
        )
        qi_guidance = st.slider(
            "Guidance Scale (CFG)", 0.0, 20.0, float(_qi_cfg.get("guidance_scale", 5.0)), 0.5, key="qi_guidance",
            help="Mức độ 'nghe lời' prompt.\n\n• ≤ 1.0: TẮT CFG — nhanh gấp đôi nhưng Negative Prompt bị BỎ QUA (chỉ dùng với chế độ nhanh Hyper-SD).\n• 5–8: chuẩn cho chế độ chất lượng — bám prompt tốt, màu tự nhiên.\n• > 10: bám prompt quá gắt → màu cháy, ảnh 'nhựa'."
        )
        qi_seed = st.number_input(
            "Seed (-1 = Random)", value=-1, key="qi_seed",
            help="Cùng seed + cùng prompt + cùng thông số = ra đúng ảnh đó lại. Đặt -1 để mỗi lần ra biến thể mới."
        )

    qi_out_dir = os.path.join(st.session_state.get("output_folder", root_dir), "ai_images")
    st.info(f"📁 Ảnh sẽ được lưu vào: `{qi_out_dir}`")

    if st.button("🎨 Tạo Ảnh", type="primary", key="qi_generate"):
        if not qi_prompt.strip():
            st.error("Vui lòng nhập prompt!")
        elif not qi_checkpoint:
            st.error("Vui lòng nhập repo HuggingFace hoặc đường dẫn model tùy chỉnh!")
        else:
            from app.services.storytelling.image_generator import StorytellingPipeline
            from app.services.storytelling.models import StoryContext

            os.makedirs(qi_out_dir, exist_ok=True)

            # Truyền checkpoint qua StoryContext — pipeline singleton tự release/reload khi đổi model
            qi_ctx = StoryContext(story_name="QuickImageGen", story_slug="quick_image_gen",
                                  genre="", checkpoint=qi_checkpoint)
            qi_pipeline = StorytellingPipeline(qi_ctx)
            if hasattr(qi_pipeline, "set_character_lora"):
                qi_pipeline.set_character_lora(None)  # tab này không dùng LoRA nhân vật

            qi_progress_text = st.empty()
            qi_progress_bar = st.progress(0)
            qi_results = []
            try:
                qi_progress_text.text("Đang tải mô hình (lần đầu có thể mất vài phút)...")
                qi_pipeline.warmup(num_steps=int(qi_steps), guidance_scale=float(qi_guidance))
                for _w in qi_pipeline.warnings:
                    st.warning(_w)

                total = int(qi_num)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                for i in range(total):
                    qi_progress_text.text(f"Đang sinh ảnh {i + 1}/{total}...")
                    seed_i = -1 if int(qi_seed) == -1 else int(qi_seed) + i
                    img, used_seed = qi_pipeline.generate_draft(
                        prompt=qi_prompt.strip(),
                        negative_prompt=qi_neg.strip(),
                        face_embedding=None,
                        face_image=None,
                        seed=seed_i,
                        width=int(qi_w),
                        height=int(qi_h),
                        num_steps=int(qi_steps),
                        guidance_scale=float(qi_guidance),
                    )
                    out_path = os.path.join(qi_out_dir, f"img_{stamp}_{i + 1}_seed{used_seed}.png")
                    img.save(out_path)
                    qi_results.append({"path": out_path, "seed": used_seed})
                    qi_progress_bar.progress((i + 1) / total)
            except Exception as ex:
                import traceback
                logger.error(f"Lỗi Workflow 7 (Tạo Ảnh AI): {ex}")
                logger.error(traceback.format_exc())
                st.error(f"Lỗi sinh ảnh: {ex}")
            finally:
                try:
                    qi_pipeline.free_vram_cache()
                except Exception:
                    pass

            if qi_results:
                st.session_state["qi_results"] = qi_results
                qi_progress_text.text("")

    # Kết quả giữ trong session_state để không mất khi Streamlit rerun
    if st.session_state.get("qi_results"):
        saved_results = [r for r in st.session_state["qi_results"] if os.path.exists(r["path"])]
        if saved_results:
            st.success(f"✅ Đã tạo {len(saved_results)} ảnh!")
            qi_cols = st.columns(3)
            for idx, r in enumerate(saved_results):
                with qi_cols[idx % 3]:
                    st.image(r["path"], caption=f"Seed: {r['seed']}")
                    with open(r["path"], "rb") as f:
                        st.download_button(
                            label="📥 Tải xuống",
                            data=f.read(),
                            file_name=os.path.basename(r["path"]),
                            mime="image/png",
                            key=f"qi_dl_{idx}"
                        )



