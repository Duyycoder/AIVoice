# -*- coding: utf-8 -*-
"""Pipeline sinh video hàng loạt với resume và error handling.

Bỏ N cặp (md + audio [+ srt]) vào 1 thư mục → nhận N video.
- State: batch_state.json trong output dir → resume item dở
- Error handling: exception 1 item → ghi failed, tiếp item sau
- torch.cuda.OutOfMemoryError → release() + retry 1 lần
"""
import os
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable
from loguru import logger


@dataclass
class BatchItem:
    stem: str
    md_path: str
    audio_path: str
    srt_path: str = ""


@dataclass
class BatchReport:
    items: List[dict] = field(default_factory=list)
    # Per-item: {stem, status, n_scenes, scenes_no_identity, video_path, error, time_seconds}

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"items": self.items}, f, ensure_ascii=False, indent=2)


# Trạng thái hợp lệ cho mỗi item
STATUS_PENDING = "pending"
STATUS_SCRIPT_DONE = "script_done"
STATUS_IMAGES_DONE = "images_done"
STATUS_UPSCALE_DONE = "upscale_done"
STATUS_VIDEO_DONE = "video_done"
STATUS_FAILED = "failed"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def scan_batch_dir(batch_dir: str) -> List[BatchItem]:
    """Scan thư mục → ghép md+audio theo stem.
    
    Hỗ trợ 2 layout:
    1. Flat: ep01.md + ep01.wav cùng thư mục
    2. Subdirs: ep01/script.md + ep01/audio.wav
    """
    items = []
    if not os.path.isdir(batch_dir):
        return items

    # Layout 1: flat files
    md_files = {}
    audio_files = {}
    srt_files = {}

    for f in os.listdir(batch_dir):
        fpath = os.path.join(batch_dir, f)
        if not os.path.isfile(fpath):
            continue
        stem, ext = os.path.splitext(f)
        ext_lower = ext.lower()
        if ext_lower == ".md":
            if " - [VI] " in f:
                md_files[stem] = fpath
        elif ext_lower in AUDIO_EXTENSIONS:
            audio_files[stem] = fpath
        elif ext_lower == ".srt":
            srt_files[stem] = fpath

    for stem, md_path in sorted(md_files.items()):
        if stem in audio_files:
            items.append(BatchItem(
                stem=stem,
                md_path=md_path,
                audio_path=audio_files[stem],
                srt_path=srt_files.get(stem, ""),
            ))

    # Layout 2: subdirectories (nếu chưa tìm thấy flat)
    if not items:
        for d in sorted(os.listdir(batch_dir)):
            subdir = os.path.join(batch_dir, d)
            if not os.path.isdir(subdir):
                continue
            md_path = ""
            audio_path = ""
            srt_path = ""
            for f in os.listdir(subdir):
                fpath = os.path.join(subdir, f)
                ext_lower = os.path.splitext(f)[1].lower()
                if ext_lower == ".md" and not md_path:
                    md_path = fpath
                elif ext_lower in AUDIO_EXTENSIONS and not audio_path:
                    audio_path = fpath
                elif ext_lower == ".srt" and not srt_path:
                    srt_path = fpath
            if md_path and audio_path:
                items.append(BatchItem(
                    stem=d,
                    md_path=md_path,
                    audio_path=audio_path,
                    srt_path=srt_path,
                ))

    return items


class _BatchState:
    """Quản lý batch_state.json trong output dir — resume item dở."""

    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "batch_state.json")
        self.data = {"items": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"items": {}}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_item_status(self, stem: str) -> str:
        return self.data["items"].get(stem, {}).get("status", STATUS_PENDING)

    def get_item_task_dir(self, stem: str) -> str:
        return self.data["items"].get(stem, {}).get("task_dir", "")

    def update_item(self, stem: str, status: str, task_dir: str = "",
                    error: str = None):
        if stem not in self.data["items"]:
            self.data["items"][stem] = {}
        self.data["items"][stem]["status"] = status
        if task_dir:
            self.data["items"][stem]["task_dir"] = task_dir
        if error is not None:
            self.data["items"][stem]["error"] = error
        self.save()


def run_batch(
    ctx_mgr,
    items: List[BatchItem],
    output_dir: str,
    options: Optional[dict] = None,
    progress_cb: Optional[Callable] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> BatchReport:
    """Chạy tuần tự từng item: step1 → step2 → step3 → assemble.

    FIX Gap4: Mỗi item có task_dir riêng + state_path riêng trong task_dir/state.json.
    """
    if options is None:
        options = {}

    os.makedirs(output_dir, exist_ok=True)
    state = _BatchState(output_dir)
    report = BatchReport()

    use_semantic_split = options.get("use_semantic_split", True)
    # Chính sách 06/07: KHÔNG dùng Whisper trong chế độ tự động — có SRT thì dùng,
    # không có thì tự tạo SRT từ kịch bản (M1).
    use_whisper = options.get("use_whisper", False)
    enable_upscale = options.get("enable_upscale", True)

    total_items = len(items)

    for item_idx, item in enumerate(items):
        if stop_flag and stop_flag():
            logger.info("[BatchRunner] Stop flag set — dừng sau item trước.")
            break

        item_status = state.get_item_status(item.stem)

        # Skip item đã hoàn thành
        if item_status == STATUS_VIDEO_DONE:
            logger.info(f"[BatchRunner] Skip {item.stem}: đã done.")
            report.items.append({
                "stem": item.stem, "status": STATUS_VIDEO_DONE,
                "video_path": "", "error": None, "time_seconds": 0,
            })
            continue

        if progress_cb:
            progress_cb(
                f"[{item_idx + 1}/{total_items}] {item.stem}",
                int(100 * item_idx / max(total_items, 1))
            )

        start_time = time.time()
        try:
            _process_single_item(
                ctx_mgr, item, output_dir, state, options,
                use_semantic_split, use_whisper, enable_upscale, progress_cb,
            )
            elapsed = time.time() - start_time
            report.items.append({
                "stem": item.stem,
                "status": STATUS_VIDEO_DONE,
                "video_path": os.path.join(output_dir, f"{item.stem}.mp4"),
                "error": None,
                "time_seconds": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)

            # Retry OOM 1 lần
            is_oom = "OutOfMemoryError" in error_msg or "CUDA out of memory" in error_msg
            if is_oom:
                logger.warning(f"[BatchRunner] OOM cho {item.stem}, retry 1 lần...")
                try:
                    from app.services.storytelling.image_generator import StorytellingPipeline
                    StorytellingPipeline().release()
                except Exception:
                    pass

                try:
                    _process_single_item(
                        ctx_mgr, item, output_dir, state, options,
                        use_semantic_split, use_whisper, enable_upscale, progress_cb,
                    )
                    elapsed = time.time() - start_time
                    report.items.append({
                        "stem": item.stem,
                        "status": STATUS_VIDEO_DONE,
                        "video_path": os.path.join(output_dir, f"{item.stem}.mp4"),
                        "error": None,
                        "time_seconds": round(elapsed, 1),
                    })
                    continue
                except Exception as e2:
                    error_msg = str(e2)

            # Ghi failed, tiếp item sau
            state.update_item(item.stem, STATUS_FAILED, error=error_msg)
            report.items.append({
                "stem": item.stem,
                "status": STATUS_FAILED,
                "video_path": "",
                "error": error_msg,
                "time_seconds": round(elapsed, 1),
            })
            logger.error(f"[BatchRunner] FAILED {item.stem}: {error_msg}")

    # Ghi report
    report_path = os.path.join(output_dir, "batch_report.json")
    report.to_json(report_path)
    logger.info(f"[BatchRunner] Done. Report: {report_path}")
    return report


def _process_single_item(
    ctx_mgr, item, output_dir, state, options,
    use_semantic_split, use_whisper, enable_upscale, progress_cb,
):
    """Xử lý 1 item từ bước đang dở (resume) hoặc từ đầu."""
    from app.services.storytelling.orchestrator import StorytellingOrchestrator
    from app.services.storytelling.video_assembler import assemble_video, FrameInfo

    _mc_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    current_status = state.get_item_status(item.stem)

    # Tạo task_dir riêng cho item (FIX Gap4)
    existing_task_dir = state.get_item_task_dir(item.stem)
    if existing_task_dir and os.path.isdir(existing_task_dir):
        task_dir = existing_task_dir
    else:
        task_id = str(uuid.uuid4())
        _storage_env = os.getenv("MC_STORAGE_TASKS")
        if _storage_env:
            task_dir = os.path.join(os.path.abspath(_storage_env), "video", "tasks", f"batch_{task_id}")
        else:
            task_dir = os.path.join(_mc_root, "storage", "tasks", f"batch_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

    # State path riêng cho item (FIX Gap4)
    item_state_path = os.path.join(task_dir, "state.json")

    orchestrator = StorytellingOrchestrator(ctx_mgr)
    orchestrator._state_path_override = item_state_path

    # --- Step 1: Script ---
    if current_status in (STATUS_PENDING, ""):
        def step1_prog(msg, pct):
            if progress_cb:
                progress_cb(f"  [{item.stem}] {msg}", pct)

        scenes, task_dir = orchestrator.step1_generate_script(
            md_path=item.md_path,
            audio_path=item.audio_path,
            srt_path=item.srt_path,
            progress_callback=step1_prog,
            use_whisper=use_whisper,
            use_semantic_split=use_semantic_split,
            state_path_override=item_state_path,
        )
        state.update_item(item.stem, STATUS_SCRIPT_DONE, task_dir=task_dir)
        current_status = STATUS_SCRIPT_DONE
    else:
        # Load state từ bước dở
        loaded = orchestrator.load_state()
        if loaded:
            from app.services.storytelling.models import scene_from_dict
            scenes = [scene_from_dict(s) for s in loaded.get("scenes", [])]
            task_dir = loaded.get("task_dir", task_dir)
        else:
            raise RuntimeError(f"Không thể load state cho {item.stem}")

    # --- Step 2: Generate Images ---
    if current_status in (STATUS_SCRIPT_DONE,):
        def step2_prog(msg, pct):
            if progress_cb:
                progress_cb(f"  [{item.stem}] {msg}", pct)

        orchestrator.step2_generate_images(
            scenes=scenes,
            task_dir=task_dir,
            progress_callback=step2_prog,
        )
        state.update_item(item.stem, STATUS_IMAGES_DONE, task_dir=task_dir)
        current_status = STATUS_IMAGES_DONE

    # --- Step 3 + Assemble: dùng step3_render_final của orchestrator ---
    # M2 FIX: bản cũ gọi orchestrator.step3_upscale (KHÔNG TỒN TẠI → AttributeError)
    # rồi tự ghép video với FrameInfo(path=, duration=) (SAI tên field → TypeError)
    # và bỏ qua SRT sinh từ kịch bản. step3_render_final làm đúng cả 3 việc:
    # upscale (postprocess) + burn phụ đề + ghép audio/BGM.
    if current_status in (STATUS_IMAGES_DONE, STATUS_UPSCALE_DONE):
        def step3_prog(msg, pct):
            if progress_cb:
                progress_cb(f"  [{item.stem}] {msg}", pct)

        # SRT ưu tiên từ state của item (bao gồm generated_from_script.srt của M1)
        item_state = orchestrator.load_state() or {}
        srt_for_video = item_state.get("srt_path", "") or item.srt_path
        if srt_for_video and not os.path.exists(srt_for_video):
            srt_for_video = ""

        final_video = orchestrator.step3_render_final(
            scenes=scenes,
            task_dir=task_dir,
            audio_path=item.audio_path,
            srt_path=srt_for_video,
            bgm_path=options.get("bgm_path", ""),
            bgm_volume=options.get("bgm_volume", 0.15),
            enable_upscaling=enable_upscale,
            burn_subtitles=options.get("burn_subtitles", True),
            progress_callback=step3_prog,
        )

        # Copy video ra output_dir với tên theo stem
        output_path = os.path.join(output_dir, f"{item.stem}.mp4")
        import shutil
        shutil.copy2(final_video, output_path)

        state.update_item(item.stem, STATUS_VIDEO_DONE, task_dir=task_dir)
        logger.info(f"[BatchRunner] Video done: {output_path}")
