# -*- coding: utf-8 -*-
"""Tải Tây Du Ký (西遊記) bản public domain và tách thành chương .md cho pipeline.

Nguồn: Project Gutenberg ebook #23962 — nguyên tác chữ Hán của Ngô Thừa Ân
(khoảng 1592), đã thuộc phạm vi công cộng. Tệp Gutenberg cũng ghi rõ giấy phép
ở phần đầu/cuối, script này cắt bỏ hai phần đó và chỉ giữ nội dung tác phẩm.

Vì sao lấy nguyên tác chữ Hán chứ không lấy bản dịch tiếng Việt có sẵn:
bản thân tác phẩm đã hết hạn bản quyền, nhưng **bản dịch thì có bản quyền riêng
của dịch giả** (các bản tiếng Việt phổ biến đều là bản dịch thập niên 1980 trở
lại đây, vẫn còn trong thời hạn bảo hộ). Lấy nguyên tác rồi để chính Bước 1 của
pipeline dịch sang tiếng Việt vừa sạch bản quyền, vừa là màn trình diễn đúng cho
tính năng dịch AI của đồ án.

Cách dùng:
    ..\\..\\.venv\\Scripts\\python.exe scripts\\fetch_jttw_chapters.py --chapters 3

Kết quả: storage/story_sources/tay_du_ky/chuong_001.md ...
Đưa thư mục đó vào Bước 1 (Nguồn truyện -> Local Folder) để dịch sang tiếng Việt.
"""
import argparse
import os
import re
import sys
import urllib.request

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC_ROOT)

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/23962/pg23962.txt",
    "https://www.gutenberg.org/files/23962/23962-0.txt",
]
START_MARK = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARK = "*** END OF THE PROJECT GUTENBERG EBOOK"
# Mốc hồi: 第一回, 第二回 ... 第一百回
CHAPTER_RE = re.compile(r"^第[〇零一二三四五六七八九十百]+回\s*(.*)$", re.MULTILINE)
UA = {"User-Agent": "MediaComposer-story-fetch/1.0 (academic use)"}


def download() -> str:
    last = None
    for url in GUTENBERG_URLS:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
            print(f"[INFO] Da tai {len(raw)//1024} KB tu {url}")
            return raw.decode("utf-8", "ignore")
        except Exception as e:
            last = e
            print(f"[WARN] {url}: {e}")
    raise SystemExit(f"[LOI] Khong tai duoc van ban: {last}")


def strip_license(text: str) -> str:
    """Bỏ phần giấy phép Gutenberg ở đầu và cuối, chỉ giữ tác phẩm."""
    start = text.find(START_MARK)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find(END_MARK)
    if end == -1:
        end = len(text)
    return text[start:end]


def split_chapters(body: str) -> list:
    """Tách theo mốc 第N回. Trả list (tieu_de, noi_dung)."""
    marks = list(CHAPTER_RE.finditer(body))
    if not marks:
        raise SystemExit("[LOI] Khong tim thay moc chuong '第N回' trong van ban.")
    chapters = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        title = " ".join(m.group(0).split())
        content = body[m.end():end].strip()
        if content:
            chapters.append((title, content))
    return chapters


def clean(content: str) -> str:
    """Nối lại dòng bị ngắt cứng, bỏ ký tự căn lề toàn giác của bản Gutenberg."""
    content = content.replace("　", " ").replace("\r\n", "\n")
    paragraphs = []
    for block in re.split(r"\n\s*\n", content):
        merged = " ".join(line.strip() for line in block.split("\n") if line.strip())
        merged = re.sub(r"\s{2,}", " ", merged).strip()
        if merged:
            paragraphs.append(merged)
    return "\n\n".join(paragraphs)


def main():
    p = argparse.ArgumentParser(description="Tai Tay Du Ky public domain -> chuong .md")
    p.add_argument("--chapters", type=int, default=3,
                   help="So hoi dau tien can xuat (0 = tat ca 100 hoi)")
    p.add_argument("--out_dir", default="")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        _MC_ROOT, "storage", "story_sources", "tay_du_ky")
    os.makedirs(out_dir, exist_ok=True)

    chapters = split_chapters(strip_license(download()))
    print(f"[INFO] Tach duoc {len(chapters)} hoi")

    limit = args.chapters if args.chapters > 0 else len(chapters)
    written = 0
    for idx, (title, content) in enumerate(chapters[:limit], start=1):
        body = clean(content)
        path = os.path.join(out_dir, f"chuong_{idx:03d}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{body}\n")
        written += 1
        print(f"  chuong_{idx:03d}.md  {len(body.split())} tu  |  {title[:44]}")

    with open(os.path.join(out_dir, "NGUON.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Nguồn dữ liệu truyện\n\n"
            "- **Tác phẩm:** 西遊記 (Tây Du Ký) — Ngô Thừa Ân, khoảng 1592\n"
            "- **Tình trạng bản quyền:** thuộc phạm vi công cộng (public domain)\n"
            "- **Nguồn tải:** Project Gutenberg ebook #23962 "
            "— https://www.gutenberg.org/ebooks/23962\n"
            "- **Xử lý:** cắt bỏ phần giấy phép Gutenberg ở đầu/cuối, tách theo mốc hồi "
            "`第N回`, nối lại các dòng bị ngắt cứng.\n"
            "- **Bản dịch tiếng Việt:** do Bước 1 của chính hệ thống sinh ra "
            "(Gemini local / Ollama), KHÔNG lấy từ bản dịch có bản quyền nào.\n")

    print(f"\n[XONG] {written} chuong -> {out_dir}")
    print("[INFO] Buoc tiep: Buoc 1 (Nguon truyen -> Local Folder) tro vao thu muc tren "
          "de dich sang tieng Viet.")


if __name__ == "__main__":
    main()
