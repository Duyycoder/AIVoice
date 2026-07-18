import json
from types import SimpleNamespace

from app.services.storytelling.batch_video_runner import (
    STATUS_UPSCALE_DONE,
    STATUS_VIDEO_DONE,
    _BatchState,
    _reconcile_video_done,
)
from adapter_video_cli import count_incomplete_items


def test_stale_video_done_recovers_final_video_from_task_dir(tmp_path):
    output_dir = tmp_path / "output"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "state.json").write_text(
        json.dumps({"step": "DONE"}), encoding="utf-8")
    payload = b"fake-mp4-payload"
    (task_dir / "final_video.mp4").write_bytes(payload)

    state = _BatchState(str(output_dir))
    item = SimpleNamespace(stem="chapter-01")
    state.update_item(item.stem, STATUS_VIDEO_DONE, task_dir=str(task_dir))

    status = _reconcile_video_done(state, item, str(output_dir))

    assert status == STATUS_VIDEO_DONE
    assert (output_dir / "chapter-01.mp4").read_bytes() == payload


def test_stale_video_done_without_any_video_is_downgraded(tmp_path):
    output_dir = tmp_path / "output"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "state.json").write_text(
        json.dumps({"step": "DONE"}), encoding="utf-8")

    state = _BatchState(str(output_dir))
    item = SimpleNamespace(stem="chapter-01")
    state.update_item(item.stem, STATUS_VIDEO_DONE, task_dir=str(task_dir))

    assert _reconcile_video_done(state, item, str(output_dir)) == STATUS_UPSCALE_DONE
    assert state.get_item_status(item.stem) == STATUS_UPSCALE_DONE


def test_adapter_counts_missing_reported_video_as_incomplete(tmp_path):
    report = SimpleNamespace(items=[{
        "status": STATUS_VIDEO_DONE,
        "video_path": str(tmp_path / "missing.mp4"),
    }])

    assert count_incomplete_items(report, total_items=1) == 1


def test_adapter_accepts_only_existing_nonempty_video(tmp_path):
    video = tmp_path / "chapter.mp4"
    video.write_bytes(b"video")
    report = SimpleNamespace(items=[{
        "status": STATUS_VIDEO_DONE,
        "video_path": str(video),
    }])

    assert count_incomplete_items(report, total_items=1) == 0
