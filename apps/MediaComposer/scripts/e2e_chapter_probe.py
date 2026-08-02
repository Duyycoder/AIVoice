# -*- coding: utf-8 -*-
"""Probe một chương Tây Du Ký từ bản dịch đến video cuối.

Luồng mặc định chạy đủ:

    dịch -> TTS -> tách cảnh -> prompt -> ảnh -> phụ đề -> ghép video

``--reuse-upstream`` dùng lại ``chuong_vi.md`` và ảnh trong ``frames/`` của lần
probe cũ. Chế độ này chủ ý chỉ bỏ qua các bước LLM/SD đắt tiền; TTS, phụ đề và
FFmpeg luôn được chạy thật để kiểm tra đúng phần cuối của pipeline.

Ví dụ probe nhỏ nhất, dùng fixture Tây Du Ký đã có:

    ..\\..\\.venv\\Scripts\\python.exe scripts\\e2e_chapter_probe.py ^
        --md storage\\story_sources\\tay_du_ky\\chuong_001.md ^
        --out storage\\quicktest\\e2e --reuse-upstream --scenes 1 ^
        --probe-chars 240
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AIVOICE_ROOT = os.path.abspath(os.path.join(_MC_ROOT, "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_AIVOICE_ROOT, ".."))
if _MC_ROOT not in sys.path:
    sys.path.insert(0, _MC_ROOT)


def load_global_config() -> dict:
    """Đọc config hợp nhất của ứng dụng tổng."""
    path = os.path.join(_REPO_ROOT, "configs", "global_config.json")
    if not os.path.exists(path):
        raise SystemExit(f"[LOI] Khong thay {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_llm_credentials(global_config: dict | None = None):
    """Đọc khoá + endpoint LLM cục bộ từ global_config.json của repo tổng."""
    cfg = global_config or load_global_config()
    key = (cfg.get("api_keys") or {}).get("gemini", "")
    if not key:
        raise SystemExit("[LOI] global_config.json khong co api_keys.gemini")
    base = (cfg.get("crawler") or {}).get(
        "gemini_offline_base_url", "http://localhost:7860/v1")
    model = (cfg.get("video") or {}).get("default_llm_model", "gemini-3-flash")
    return key, base, model


def translate_to_vietnamese(text: str, chunk_chars: int = 2500) -> str:
    """Dịch Hán -> Việt qua chính LLM cục bộ mà pipeline dùng, chia khúc theo đoạn."""
    from app.services.llm import get_llm_client
    client, model = get_llm_client()

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for paragraph in paragraphs:
        if buf and len(buf) + len(paragraph) > chunk_chars:
            chunks.append(buf)
            buf = paragraph
        else:
            buf = f"{buf}\n\n{paragraph}" if buf else paragraph
    if buf:
        chunks.append(buf)

    system = (
        "Bạn là dịch giả văn học Trung - Việt. Dịch đoạn văn cổ văn sau sang "
        "tiếng Việt hiện đại, giữ nguyên nghĩa và mạch truyện, văn phong kể "
        "chuyện tự nhiên để đọc thành lời. Chỉ trả về bản dịch, không chú thích, "
        "không thêm tiêu đề.")

    output = []
    for index, chunk in enumerate(chunks, 1):
        started = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        output.append(response.choices[0].message.content.strip())
        print(
            f"  [dich] khuc {index}/{len(chunks)} ({len(chunk)} ky tu) "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
    return "\n\n".join(output)


def limit_probe_text(text: str, max_chars: int) -> str:
    """Giới hạn fixture nhanh tại biên câu/từ, không cắt giữa một từ Unicode."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized

    prefix = normalized[:max_chars]
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?;。！？；…](?:[\"'”’)\]]*)", prefix)
    ]
    if sentence_ends:
        cut = sentence_ends[-1]
    else:
        whitespace = [match.start() for match in re.finditer(r"\s+", prefix)]
        cut = whitespace[-1] if whitespace else max_chars
    return prefix[:cut].rstrip()


def build_tts_command(
    input_path: str,
    output_path: str,
    tts_config: dict,
    *,
    python_exe: str | None = None,
) -> list[str]:
    """Tạo lệnh adapter TTS với mọi lựa chọn quan trọng được truyền tường minh."""
    engine = str(tts_config.get("engine") or "kokoro")
    voice = str(tts_config.get("voice") or "thanh_dat")
    device = str(tts_config.get("device") or "cuda")
    command = [
        python_exe or sys.executable,
        os.path.join(_AIVOICE_ROOT, "adapter_tts_cli.py"),
        "--input", os.path.abspath(input_path),
        "--output", os.path.abspath(output_path),
        "--engine", engine,
        "--voice", voice,
        "--device", device,
        "--speed", str(float(tts_config.get("speed", 1.0))),
        "--target-lufs", str(float(tts_config.get("target_lufs", -14.0))),
        "--fade-in", str(float(tts_config.get("fade_in", 0.1))),
        "--fade-out", str(float(tts_config.get("fade_out", 0.1))),
        "--silence-duration", str(float(tts_config.get("silence_duration", 0.3))),
        "--no-cache",
    ]
    command.append("--normalize" if tts_config.get("normalize", True) else "--no-normalize")
    return command


def synthesize_narration(
    narration_path: str,
    audio_path: str,
    tts_config: dict,
) -> float:
    """Chạy TTS trong process riêng rồi trả thời lượng WAV thật."""
    command = build_tts_command(narration_path, audio_path, tts_config)
    print(
        f"      engine={tts_config.get('engine')} voice={tts_config.get('voice')} "
        f"device={tts_config.get('device')}",
        flush=True,
    )
    subprocess.run(command, cwd=_AIVOICE_ROOT, check=True)
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError(f"TTS không tạo được audio: {audio_path}")

    from app.services.storytelling.audio_utils import get_audio_duration
    return get_audio_duration(audio_path)


def _new_scene(scene_id: int, text: str, metadata: dict | None = None):
    from app.services.storytelling.models import Scene

    scene = Scene(
        scene_id=scene_id,
        text_vi=text,
        word_count=len(text.split()),
        start_time=0.0,
        end_time=0.0,
        duration_sec=0.0,
        image_prompt="",
        characters_in_scene=[],
        primary_character="",
        fallback_level=0,
        accepted_seed=-1,
        frame_path="",
    )
    scene._semantic_meta = metadata or {
        "location": "",
        "action": "",
        "summary": text[:160],
        "time_of_day": "",
    }
    return scene


def split_text_for_scenes(text: str, max_scenes: int) -> list[str]:
    """Chia toàn bộ narration thành tối đa N khối liên tiếp, không bỏ chữ."""
    max_scenes = max(1, max_scenes)
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;。！？；…])\s+", text.strip())
        if part.strip()
    ]
    if not sentences or max_scenes == 1:
        return [text.strip()]

    group_count = min(max_scenes, len(sentences))
    total_words = sum(max(len(sentence.split()), 1) for sentence in sentences)
    target_words = max(total_words / group_count, 1.0)
    groups, current = [], []
    current_words = 0
    for sentence in sentences:
        remaining_sentences = len(sentences) - sum(len(group) for group in groups) - len(current)
        remaining_groups = group_count - len(groups)
        sentence_words = max(len(sentence.split()), 1)
        should_close = (
            current
            and current_words + sentence_words > target_words
            and remaining_sentences >= remaining_groups
        )
        if should_close:
            groups.append(current)
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        groups.append(current)

    while len(groups) > group_count:
        groups[-2].extend(groups[-1])
        groups.pop()
    return [" ".join(group).strip() for group in groups if group]


def scenes_from_text(text: str, max_scenes: int):
    """Tạo Scene nhẹ cho chế độ reuse, vẫn bao phủ đúng toàn bộ narration."""
    return [
        _new_scene(index, chunk)
        for index, chunk in enumerate(split_text_for_scenes(text, max_scenes))
    ]


def scenes_from_semantic(semantic_scenes: list, max_scenes: int):
    """Coalesce cảnh semantic để probe có thể render ít ảnh nhưng không mất narration."""
    if not semantic_scenes:
        return []
    max_scenes = max(1, min(max_scenes, len(semantic_scenes)))
    if len(semantic_scenes) <= max_scenes:
        groups = [[scene] for scene in semantic_scenes]
    else:
        groups = [[] for _ in range(max_scenes)]
        for index, scene in enumerate(semantic_scenes):
            group_index = min(index * max_scenes // len(semantic_scenes), max_scenes - 1)
            groups[group_index].append(scene)

    scenes = []
    for index, group in enumerate(groups):
        text = "\n\n".join(item.text_vi.strip() for item in group if item.text_vi.strip())
        first, last = group[0], group[-1]
        metadata = {
            "location": first.location,
            "action": " / ".join(
                item.action for item in group if item.action
            )[:500],
            "summary": " ".join(
                item.summary_vi for item in group if item.summary_vi
            )[:500],
            "time_of_day": first.time_of_day or last.time_of_day,
        }
        scene = _new_scene(index, text, metadata)
        scene.characters_in_scene = list(dict.fromkeys(
            character
            for item in group
            for character in (item.characters or [])
        ))
        scenes.append(scene)
    return scenes


def assign_reused_frames(scenes: list, frames_dir: str) -> None:
    """Gắn ảnh fixture cũ theo thứ tự; báo lỗi rõ nếu không đủ ảnh."""
    frame_paths = sorted(Path(frames_dir).glob("scene_*.png"))
    if len(frame_paths) < len(scenes):
        raise RuntimeError(
            f"Fixture chỉ có {len(frame_paths)} ảnh nhưng cần {len(scenes)}: {frames_dir}")
    for scene, frame_path in zip(scenes, frame_paths):
        scene.frame_path = str(frame_path.resolve())


def validate_video(output_path: str, expected_duration: float) -> dict:
    """Decode toàn bộ output và xác nhận có cả stream video lẫn audio."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    inspect = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", os.path.abspath(output_path)],
        capture_output=True,
        text=True,
        errors="ignore",
    )
    details = inspect.stderr or ""
    if "Video:" not in details or "Audio:" not in details:
        raise RuntimeError("Video cuối thiếu stream video hoặc audio.")

    decode = subprocess.run(
        [
            ffmpeg_exe,
            "-v", "error",
            "-i", os.path.abspath(output_path),
            "-f", "null",
            "-",
        ],
        capture_output=True,
        text=True,
        errors="ignore",
    )
    if decode.returncode != 0:
        raise RuntimeError(f"Không decode được video cuối: {decode.stderr.strip()}")

    duration_match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        details,
    )
    if not duration_match:
        raise RuntimeError("FFmpeg không đọc được duration của video cuối.")
    hours, minutes, seconds = duration_match.groups()
    actual_duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    tolerance = max(0.15, 1.0 / 24.0)
    if abs(actual_duration - expected_duration) > tolerance:
        raise RuntimeError(
            f"Duration video {actual_duration:.3f}s lệch audio "
            f"{expected_duration:.3f}s quá {tolerance:.3f}s.")
    return {
        "path": os.path.abspath(output_path),
        "bytes": os.path.getsize(output_path),
        "duration_sec": actual_duration,
        "has_video": True,
        "has_audio": True,
        "decoded": True,
    }


def _configure_context():
    from app.services.storytelling.context_manager import ContextManager

    context_manager = ContextManager("tay_du_ky_probe")
    try:
        context = context_manager.load_context()
    except FileNotFoundError:
        context = context_manager.create_context("Tay Du Ky", "tien_hiep")

    style_source = os.path.join(
        _MC_ROOT, "resource", "image_presets", "thuy_mac.txt")
    with open(style_source, encoding="utf-8") as handle:
        style_text = handle.read()
    with open(context_manager.style_file, "w", encoding="utf-8") as handle:
        handle.write(style_text)
    context._style_prompt_path = context_manager.style_file
    return context


def _load_or_translate(args, output_dir: str) -> str:
    translated_path = os.path.join(output_dir, "chuong_vi.md")
    if args.reuse_upstream:
        if not os.path.isfile(translated_path):
            raise RuntimeError(
                f"--reuse-upstream cần bản dịch fixture: {translated_path}")
        with open(translated_path, encoding="utf-8-sig") as handle:
            translated = handle.read()
        print(f"[1/7] DUNG LAI {translated_path} ({len(translated)} ky tu)", flush=True)
        return translated

    import codecs
    with codecs.open(args.md, "r", "utf-8-sig") as handle:
        raw = handle.read()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    body = "\n\n".join(
        block
        for block in raw.split("\n\n")
        if block.strip() and not block.startswith("# ")
    )
    if not body.strip():
        raise RuntimeError(f"Khong doc duoc noi dung tu {args.md}")

    print(f"\n[1/7] Dich {len(body)} ky tu Han -> Viet...", flush=True)
    started = time.time()
    translated = translate_to_vietnamese(body)
    with open(translated_path, "w", encoding="utf-8") as handle:
        handle.write(translated)
    print(
        f"[1/7] XONG {time.time() - started:.1f}s -> "
        f"{translated_path} ({len(translated)} ky tu)",
        flush=True,
    )
    return translated


def _generate_images(scenes: list, context, output_dir: str, st_config: dict) -> None:
    from app.services.llm import unload_local_llm
    from app.services.storytelling.image_generator import StorytellingPipeline

    unload_local_llm()
    pipeline = StorytellingPipeline(context)
    pipeline.warmup()
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    width = st_config.get("image_width", 768)
    height = st_config.get("image_height", 432)
    shot_map = {
        "close": (576, 704),
        "medium": (704, 528),
        "wide": (width, height),
    }
    try:
        for index, scene in enumerate(scenes):
            scene_width, scene_height = shot_map.get(
                getattr(scene, "shot_type", "wide"), (width, height))
            started = time.time()
            image, seed = pipeline.generate_draft(
                prompt=scene.image_prompt,
                negative_prompt=context.get_negative_prompt(),
                face_embedding=None,
                face_image=None,
                seed=-1,
                width=scene_width,
                height=scene_height,
            )
            path = os.path.join(frames_dir, f"scene_{index:03d}.png")
            image.save(path)
            scene.frame_path = path
            scene.accepted_seed = seed
            print(
                f"      canh {index + 1}/{len(scenes)} "
                f"{scene_width}x{scene_height} {time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        pipeline.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="File .md nguyen tac chu Han")
    parser.add_argument(
        "--scenes", type=int, default=8,
        help="So anh toi da; canh duoc coalesce de van bao phu narration")
    parser.add_argument(
        "--duration", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--reuse-upstream", action="store_true",
        help="Dung lai chuong_vi.md + frames, chi chay that TTS/SRT/FFmpeg")
    parser.add_argument(
        "--probe-chars", type=int, default=0,
        help="Gioi han narration cho smoke nhanh; 0 = ca chuong")
    parser.add_argument("--tts-engine", default="")
    parser.add_argument("--tts-voice", default="")
    parser.add_argument("--tts-device", choices=["cpu", "cuda"], default="")
    args = parser.parse_args()
    if args.scenes < 1:
        parser.error("--scenes phải >= 1")
    if args.probe_chars < 0:
        parser.error("--probe-chars phải >= 0")

    output_dir = os.path.abspath(
        args.out or os.path.join(_MC_ROOT, "storage", "quicktest", "e2e"))
    os.makedirs(output_dir, exist_ok=True)

    global_config = load_global_config()
    key, base, model = load_llm_credentials(global_config)
    from app.config import config, load_storytelling_config
    config.app["llm_api_key"] = key
    config.app["llm_base_url"] = base
    config.app["llm_model"] = model
    print(f"[INFO] LLM: {model} @ {base}", flush=True)

    st_config = load_storytelling_config()
    print(
        f"[INFO] render_mode={st_config.get('render_mode')} "
        f"steps={st_config.get('num_inference_steps')} "
        f"cfg={st_config.get('guidance_scale')} "
        f"style_lora={st_config.get('style_lora')}"
        f"@{st_config.get('style_lora_weight')}",
        flush=True,
    )

    translated = _load_or_translate(args, output_dir)
    narration = limit_probe_text(translated, args.probe_chars)
    if not narration:
        raise SystemExit("[LOI] Narration rong sau khi gioi han probe.")
    narration_path = os.path.join(output_dir, "narration.md")
    with open(narration_path, "w", encoding="utf-8") as handle:
        handle.write(narration)
    if len(narration) != len(translated.strip()):
        print(
            f"      smoke fixture: {len(narration)}/{len(translated.strip())} ky tu",
            flush=True,
        )

    print("\n[2/7] TTS narration...", flush=True)
    tts_config = dict(global_config.get("tts") or {})
    tts_config["engine"] = args.tts_engine or tts_config.get("default_engine", "kokoro")
    default_voice = (
        tts_config.get(f"{tts_config['engine']}_voice")
        or tts_config.get("default_voice")
        or "thanh_dat"
    )
    tts_config["voice"] = args.tts_voice or default_voice
    tts_config["device"] = args.tts_device or tts_config.get("device", "cuda")
    audio_path = os.path.join(output_dir, "narration.wav")
    started = time.time()
    audio_duration = synthesize_narration(narration_path, audio_path, tts_config)
    print(
        f"[2/7] XONG {time.time() - started:.1f}s -> "
        f"{audio_path} ({audio_duration:.3f}s)",
        flush=True,
    )

    context = _configure_context()
    if args.reuse_upstream:
        print("\n[3/7] DUNG LAI scene fixture (khong goi semantic LLM).", flush=True)
        scenes = scenes_from_text(narration, args.scenes)
        print("[4/7] DUNG LAI prompt fixture.", flush=True)
        prompts_path = os.path.join(output_dir, "prompts.json")
        if os.path.isfile(prompts_path):
            with open(prompts_path, encoding="utf-8") as handle:
                prompts = json.load(handle)
            for scene, prompt in zip(scenes, prompts):
                scene.image_prompt = str(prompt.get("prompt") or "")
                scene.shot_type = str(prompt.get("shot") or "wide")
                scene._llm_action = str(prompt.get("action") or "")
        print("[5/7] DUNG LAI anh fixture.", flush=True)
        assign_reused_frames(scenes, os.path.join(output_dir, "frames"))
    else:
        print(
            f"\n[3/7] Tach canh ngu nghia (duration that = {audio_duration:.3f}s)...",
            flush=True,
        )
        started = time.time()
        from app.services.storytelling.semantic_scene_splitter import split_scenes_semantic
        semantic_scenes = split_scenes_semantic(narration, audio_duration)
        if not semantic_scenes:
            raise RuntimeError(
                "Tach canh tra None — LLM loi hoac validation that bai.")
        scenes = scenes_from_semantic(semantic_scenes, args.scenes)
        print(
            f"[3/7] XONG {time.time() - started:.1f}s -> "
            f"{len(semantic_scenes)} canh semantic, coalesce con {len(scenes)}",
            flush=True,
        )

        print("\n[4/7] Sinh prompt (style lock + action)...", flush=True)
        started = time.time()
        from app.services.storytelling.llm_prompter import generate_prompts_batch
        generate_prompts_batch(scenes, context)
        print(f"[4/7] XONG {time.time() - started:.1f}s", flush=True)

        print(f"\n[5/7] Sinh {len(scenes)} anh (classic, style lock)...", flush=True)
        started = time.time()
        _generate_images(scenes, context, output_dir, st_config)
        print(f"[5/7] XONG {time.time() - started:.1f}s", flush=True)

    print("\n[6/7] Map timeline + tao subtitle tu narration...", flush=True)
    from app.services.storytelling.srt_mapper import (
        generate_srt_from_scenes,
        map_semantic_scenes_to_srt,
    )
    scenes = map_semantic_scenes_to_srt(scenes, [], audio_duration)
    subtitle_path = generate_srt_from_scenes(
        scenes, os.path.join(output_dir, "subtitle.srt"))
    print(
        f"[6/7] XONG -> {subtitle_path} "
        f"({scenes[0].start_time:.3f}s..{scenes[-1].end_time:.3f}s)",
        flush=True,
    )

    print("\n[7/7] Ghep video + burn subtitle...", flush=True)
    from app.services.storytelling.video_assembler import FrameInfo, assemble_video
    output_path = os.path.join(output_dir, "final_video.mp4")
    frames = [
        FrameInfo(frame_path=scene.frame_path, duration_sec=scene.duration_sec)
        for scene in scenes
    ]
    started = time.time()
    assemble_video(
        frames=frames,
        audio_path=audio_path,
        srt_path=subtitle_path,
        output_path=output_path,
        burn_subtitles=True,
    )
    validation = validate_video(output_path, audio_duration)
    print(
        f"[7/7] XONG {time.time() - started:.1f}s -> {output_path} "
        f"({validation['bytes']} bytes, {validation['duration_sec']:.3f}s)",
        flush=True,
    )

    manifest = {
        "source_md": os.path.abspath(args.md),
        "reuse_upstream": args.reuse_upstream,
        "probe_chars": args.probe_chars,
        "narration_chars": len(narration),
        "tts": {
            "engine": tts_config["engine"],
            "voice": tts_config["voice"],
            "device": tts_config["device"],
        },
        "audio_path": os.path.abspath(audio_path),
        "audio_duration_sec": audio_duration,
        "subtitle_path": os.path.abspath(subtitle_path),
        "scene_count": len(scenes),
        "video": validation,
    }
    manifest_path = os.path.join(output_dir, "probe_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"\n[XONG] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
