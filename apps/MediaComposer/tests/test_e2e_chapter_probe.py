# -*- coding: utf-8 -*-
"""Regressions for the downstream half of the Tây Du Ký chapter probe."""
import importlib.util
import math
import os
import sys
import wave
from array import array
from pathlib import Path

from PIL import Image


_MC_ROOT = Path(__file__).resolve().parents[1]
if str(_MC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MC_ROOT))


def _load_probe_module():
    path = _MC_ROOT / "scripts" / "e2e_chapter_probe.py"
    spec = importlib.util.spec_from_file_location("e2e_chapter_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tone(path: Path, duration_sec: float = 0.6, sample_rate: int = 24000):
    samples = array(
        "h",
        (
            int(1200 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(round(duration_sec * sample_rate))
        ),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_probe_text_split_preserves_complete_narration():
    probe = _load_probe_module()
    text = "Câu thứ nhất. Câu thứ hai dài hơn một chút! Câu thứ ba kết thúc."

    chunks = probe.split_text_for_scenes(text, 2)

    assert len(chunks) == 2
    assert " ".join(chunks) == text
    assert probe.limit_probe_text("Xin chào thế giới không bị cắt từ", 18) == "Xin chào thế giới"


def test_tts_command_is_explicit_and_disables_semantic_cache(tmp_path):
    probe = _load_probe_module()
    command = probe.build_tts_command(
        str(tmp_path / "chapter.md"),
        str(tmp_path / "chapter.wav"),
        {
            "engine": "kokoro",
            "voice": "thanh_dat",
            "device": "cpu",
            "normalize": False,
        },
        python_exe="python-for-test",
    )

    assert command[0] == "python-for-test"
    assert command[1].endswith(os.path.join("AIVoice", "adapter_tts_cli.py"))
    assert command[command.index("--engine") + 1] == "kokoro"
    assert command[command.index("--voice") + 1] == "thanh_dat"
    assert command[command.index("--device") + 1] == "cpu"
    assert "--no-cache" in command
    assert "--no-normalize" in command


def test_ffmpeg_filter_path_escapes_windows_drive_colon():
    from app.services.storytelling.video_assembler import _escape_filter_path

    assert _escape_filter_path(r"F:\media\fonts") == r"F\:/media/fonts"


def test_generated_subtitle_is_burned_into_decodable_audio_video(
    tmp_path,
    monkeypatch,
):
    probe = _load_probe_module()
    from app.services.storytelling import video_assembler
    from app.services.storytelling.srt_mapper import (
        generate_srt_from_scenes,
        map_semantic_scenes_to_srt,
    )

    frame_path = tmp_path / "scene.png"
    audio_path = tmp_path / "narration.wav"
    subtitle_path = tmp_path / "subtitle.srt"
    output_path = tmp_path / "final.mp4"
    Image.new("RGB", (320, 180), (28, 44, 72)).save(frame_path)
    _write_tone(audio_path)

    scenes = probe.scenes_from_text(
        "Tôn Ngộ Không bay qua núi. Trư Bát Giới đi theo sau.",
        1,
    )
    scenes[0].frame_path = str(frame_path)
    scenes = map_semantic_scenes_to_srt(scenes, [], 0.6)
    generate_srt_from_scenes(scenes, str(subtitle_path))

    monkeypatch.setattr(
        video_assembler,
        "load_storytelling_config",
        lambda: {
            "output_width": 320,
            "output_height": 180,
            "video_fps": 24,
            "subtitle_font": "Arial",
            "subtitle_font_size": 18,
            "subtitle_color": "white",
            "subtitle_border": 1,
            "subtitle_position": "bottom",
        },
    )
    monkeypatch.setattr(
        video_assembler,
        "_get_video_codec_args",
        lambda: ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage"],
    )
    video_assembler.assemble_video(
        frames=[
            video_assembler.FrameInfo(
                frame_path=str(frame_path),
                duration_sec=scenes[0].duration_sec,
            ),
        ],
        audio_path=str(audio_path),
        srt_path=str(subtitle_path),
        output_path=str(output_path),
        burn_subtitles=True,
    )

    validation = probe.validate_video(str(output_path), 0.6)
    assert validation["decoded"] is True
    assert validation["has_audio"] is True
    assert validation["has_video"] is True
    assert "Tôn Ngộ Không" in subtitle_path.read_text(encoding="utf-8")
