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
    """W/H/thời lượng đọc từ ffmpeg — máy đích KHÔNG có ffprobe (CB5)."""
    from app.utils import utils
    info = utils.probe_media_info(path)
    if not info:
        log_json("download_warning", {"message": f"Không đọc được thông số video: {path}"})
        return {"width": 0, "height": 0, "duration": 0}
    return {"width": info["width"], "height": info["height"], "duration": info["duration"]}


def read_urls(args) -> list:
    """[(link, số thứ tự trong lô)] — giữ thứ tự người dùng nhập, bỏ link trùng.

    Số thứ tự đi KÈM từng link (cờ `--batch-index` lặp lại, khớp 1-1 với `--url`):
    bỏ link trùng mà không bỏ kèm số của nó là cả lô lệch chỗ khi ghép.
    """
    urls = list(args.url or [])
    indexes = list(getattr(args, "batch_index", None) or [])
    if args.urls_file and os.path.exists(args.urls_file):
        with open(args.urls_file, encoding="utf-8-sig") as fh:
            urls += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    seen, out = set(), []
    for i, u in enumerate(urls):
        if u in seen:
            continue
        seen.add(u)
        out.append((u, indexes[i] if i < len(indexes) else float(i + 1)))
    return out


def batch_indexes(base_index: float, count: int) -> list:
    """Số thứ tự trong lô cho các video giải ra từ MỘT link đứng ở chỗ `base_index`.

    Một link ra đúng một video thì giữ nguyên số của nó; playlist ra m video thì
    video con thứ j nhận `base + j/1000` — nhờ vậy cả playlist nằm gọn đúng chỗ
    người dùng đã đặt link, không đẩy lệch các mục đứng sau trong lô.
    """
    if count <= 1:
        return [base_index]
    return [base_index + (j + 1) / 1000.0 for j in range(count)]


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
    p.add_argument("--probe-file", default="", help="Chỉ đọc thông số một file video có sẵn rồi thoát")
    p.add_argument("--batch-id", default="", help="Mã lô, ghi vào video.json để bước ghép biết thứ tự")
    p.add_argument("--batch-index", action="append", type=float, default=[],
                   help="Số thứ tự trong lô của từng --url (lặp lại, khớp 1-1 theo thứ tự)")
    args = p.parse_args()

    # Chế độ đọc thông số file cục bộ: orchestrator gọi khi người dùng nhập video
    # có sẵn vào thư viện (nó không được phép tự import ffmpeg/torch).
    if args.probe_file:
        log_json("probe_file_done", probe_meta(args.probe_file))
        sys.exit(0)

    urls = read_urls(args)
    if not urls:
        log_json("batch_failed", {"error": "Chưa có link nào để tải."})
        sys.exit(1)
    if args.batch_index and len(args.batch_index) != len(args.url):
        log_json("download_warning", {"message": "Số --batch-index không khớp số --url — "
                                                 "lô có thể bị ghép sai thứ tự."})

    from app.services.video_downloader import probe_entries, download_video

    cookies = args.cookies_file or None

    # 1) Giải mọi link thành danh sách video phẳng
    entries, probe_errors = [], []
    seen_urls = set()
    for u, base_index in urls:
        try:
            found = probe_entries(u, args.platform, cookies, args.max_items)
        except Exception as e:
            probe_errors.append({"url": u, "error": str(e)})
            log_json("item_failed", {"url": u, "error": str(e)})
            if args.stop_on_error:
                log_json("batch_failed", {"error": str(e)})
                sys.exit(1)
            continue
        idxs = batch_indexes(base_index, len(found))
        for j, e in enumerate(found):
            if e["url"] and e["url"] not in seen_urls:
                seen_urls.add(e["url"])
                e["batch_index"] = idxs[j]
                e["index"] = len(entries) + 1
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
                "batch_id": args.batch_id,
                "batch_index": entry.get("batch_index", float(idx)),
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
