"""CLI cào video hàng loạt cho orchestrator (nhánh chỉ-video).

Nhận 1..n link (video lẻ, playlist, kênh, hashtag), giải ra danh sách video rồi
tải lần lượt vào THƯ VIỆN: mỗi video là một thư mục con của --output-dir chứa
`video.json` + file mp4. Orchestrator chỉ việc quét thư mục đó, không phải import
yt-dlp/torch (kiến trúc: mọi thứ nặng chạy trong AIVoice/.venv qua subprocess).

Mỗi dòng stdout là một JSON event để orchestrator stream qua SSE:
    probe_done | batch_start | item_start | download_progress |
    item_done | item_skipped | item_failed | batch_done
"""
import os
import sys
import json
import argparse
import datetime
import re
import unicodedata

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mc_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mc_root)
sys.path.insert(0, os.path.join(mc_root, "app"))


def log_json(event: str, data: dict):
    print(json.dumps({"event": event, **data}, ensure_ascii=False))
    sys.stdout.flush()


def slugify(text: str, max_len: int = 48) -> str:
    """Tên thư mục an toàn từ tiêu đề video (bỏ dấu, bỏ ký tự Windows cấm)."""
    text = (text or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len].strip("_") or "video"


def probe_meta(path: str) -> dict:
    """W/H/thời lượng bằng moviepy — máy đích KHÔNG có ffprobe (CB5)."""
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(path)
        w, h = clip.size
        dur = float(clip.duration or 0)
        clip.close()
        return {"width": int(w), "height": int(h), "duration": round(dur, 2)}
    except Exception as e:
        log_json("download_warning", {"message": f"Không đọc được thông số video ({e})."})
        return {"width": 0, "height": 0, "duration": 0}


def read_urls(args) -> list:
    urls = list(args.url or [])
    if args.urls_file and os.path.exists(args.urls_file):
        with open(args.urls_file, encoding="utf-8-sig") as fh:
            urls += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    # Giữ thứ tự người dùng nhập, bỏ trùng.
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def existing_ids(library_dir: str) -> dict:
    """{video_id: đường dẫn video.json} của các mục đã có trong thư viện."""
    found = {}
    if not os.path.isdir(library_dir):
        return found
    for name in os.listdir(library_dir):
        meta_path = os.path.join(library_dir, name, "video.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            vid = meta.get("source_id") or meta.get("id")
            if vid:
                found[str(vid)] = meta_path
        except (OSError, json.JSONDecodeError):
            continue
    return found


def main():
    p = argparse.ArgumentParser(description="Cào video hàng loạt (yt-dlp) vào thư viện")
    p.add_argument("--url", action="append", default=[], help="Link video/playlist/kênh (lặp lại được)")
    p.add_argument("--urls-file", default="", help="File .txt mỗi dòng một link")
    p.add_argument("--output-dir", required=True, help="Thư mục THƯ VIỆN (mỗi video một thư mục con)")
    p.add_argument("--platform", default="generic", help="bilibili|tiktok|douyin|youtube|generic")
    p.add_argument("--cookies-file", default=None, help="File cookies Netscape .txt")
    p.add_argument("--max-items", type=int, default=0, help="Giới hạn số video mỗi link (0 = không giới hạn)")
    p.add_argument("--skip-existing", action="store_true", default=False, help="Bỏ qua video đã có trong thư viện")
    p.add_argument("--stop-on-error", action="store_true", default=False, help="Dừng cả lô khi một video lỗi")
    p.add_argument("--probe-only", action="store_true", default=False, help="Chỉ liệt kê video, không tải")
    args = p.parse_args()

    urls = read_urls(args)
    if not urls:
        log_json("batch_failed", {"error": "Chưa có link nào để tải."})
        sys.exit(1)

    from app.services.video_downloader import probe_entries, download_video

    cookies = args.cookies_file or None

    # 1) Giải mọi link thành danh sách video phẳng
    entries, probe_errors = [], []
    seen_urls = set()
    for u in urls:
        try:
            found = probe_entries(u, args.platform, cookies, args.max_items)
        except Exception as e:
            probe_errors.append({"url": u, "error": str(e)})
            log_json("item_failed", {"url": u, "error": str(e)})
            if args.stop_on_error:
                log_json("batch_failed", {"error": str(e)})
                sys.exit(1)
            continue
        for e in found:
            if e["url"] and e["url"] not in seen_urls:
                seen_urls.add(e["url"])
                entries.append(e)

    if args.probe_only:
        log_json("probe_done", {"count": len(entries), "entries": entries, "errors": probe_errors})
        sys.exit(0 if entries else 1)

    if not entries:
        log_json("batch_failed", {"error": "Không giải được video nào từ các link đã nhập."})
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    known = existing_ids(args.output_dir) if args.skip_existing else {}

    log_json("batch_start", {"total": len(entries), "output_dir": os.path.abspath(args.output_dir)})

    ok_count = fail_count = skip_count = 0
    for idx, entry in enumerate(entries, 1):
        vid = entry["id"] or slugify(entry["title"])
        head = {"index": idx, "total": len(entries), "title": entry["title"], "url": entry["url"]}

        if args.skip_existing and vid in known:
            skip_count += 1
            log_json("item_skipped", {**head, "reason": "Đã có trong thư viện"})
            continue

        log_json("item_start", head)
        title_slug = slugify(entry["title"])
        entry_dir = os.path.join(args.output_dir, title_slug + "_" + slugify(vid, 24))
        try:
            os.makedirs(entry_dir, exist_ok=True)
            raw_path = download_video(entry["url"], entry_dir, args.platform, cookies_file=cookies)

            # Đổi tên về <slug tiêu đề>.mp4 cho dễ nhìn trong File Explorer.
            final_path = os.path.join(entry_dir, title_slug + ".mp4")
            if os.path.abspath(raw_path) != os.path.abspath(final_path):
                try:
                    os.replace(raw_path, final_path)
                except OSError:
                    final_path = raw_path

            meta = {
                "entry_id": os.path.basename(entry_dir),
                "source_id": vid,
                "title": entry["title"],
                "url": entry["url"],
                "platform": args.platform,
                "uploader": entry.get("uploader", ""),
                "file": os.path.abspath(final_path),
                "size": os.path.getsize(final_path),
                "source": "download",
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            meta.update(probe_meta(final_path))
            with open(os.path.join(entry_dir, "video.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)

            ok_count += 1
            log_json("item_done", {**head, "path": meta["file"], "entry_id": meta["entry_id"],
                                   "duration": meta["duration"]})
        except Exception as e:
            fail_count += 1
            log_json("item_failed", {**head, "error": str(e)})
            # Thư mục rỗng do tải hỏng chỉ làm bẩn thư viện.
            try:
                if os.path.isdir(entry_dir) and not os.listdir(entry_dir):
                    os.rmdir(entry_dir)
            except OSError:
                pass
            if args.stop_on_error:
                log_json("batch_done", {"ok": ok_count, "failed": fail_count,
                                        "skipped": skip_count, "stopped": True})
                sys.exit(1)

    log_json("batch_done", {"ok": ok_count, "failed": fail_count, "skipped": skip_count})
    sys.exit(0 if ok_count or skip_count else 1)


if __name__ == "__main__":
    main()
