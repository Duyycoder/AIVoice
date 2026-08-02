# -*- coding: utf-8 -*-
"""Dựng dataset train LoRA phong cách thủy mặc từ nguồn CC0 của bảo tàng.

Nguồn: The Metropolitan Museum of Art Open Access API — mọi ảnh lấy về đều được
lọc theo cờ ``isPublicDomain == True`` (Met công bố các ảnh này theo CC0). Script
ghi kèm ``manifest.json`` lưu objectID, tiêu đề, niên đại, license và link gốc của
TỪNG ảnh, để phần "nguồn dữ liệu" trong báo cáo đồ án có bằng chứng truy vết được.

Vì sao cắt ô (tile) chứ không center-crop:
tranh thủy mặc phần lớn là trục cuốn — thủ quyển ngang rất dài hoặc lập trục dọc
rất cao. Center-crop một thủ quyển 8000x600 về ô vuông 512 sẽ vứt đi 95% bức tranh
và thường chỉ còn lại khoảng giấy trắng. Cắt trượt dọc theo cạnh dài cho ra 4-8 ô
dùng được từ MỘT bức, và mỗi ô là một bố cục khác nhau — đúng thứ LoRA phong cách
cần (cùng cách vẽ, khác nội dung).

Ô nào gần như trắng trơn (khoảng trống của tranh) bị loại bằng ngưỡng độ lệch chuẩn
độ sáng — nếu không, LoRA sẽ học rằng "phong cách này = giấy trắng".

Cách dùng:
    ..\\..\\.venv\\Scripts\\python.exe scripts\\build_style_dataset.py --target 80

Kết quả: storage/style_datasets/thuy_mac/  (ảnh .png 512x512 + .txt caption + manifest.json)
Rồi train:
    ..\\..\\.venv\\Scripts\\python.exe scripts\\train_style_lora.py ^
        --style thuy_mac --images_dir storage\\style_datasets\\thuy_mac --steps 1500
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_MC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _MC_ROOT)

from PIL import Image  # noqa: E402

MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
MET_DEPT_ASIAN_ART = 6
MET_OBJECT_PAGE = "https://www.metmuseum.org/art/collection/search"

# Truy vấn phủ các thể loại thủy mặc kinh điển. Mỗi truy vấn giới hạn trong
# department "Asian Art" và bắt buộc có ảnh.
QUERIES = [
    "ink landscape", "ink on paper hanging scroll", "handscroll ink",
    "album leaf ink", "bamboo ink painting", "plum blossom ink",
    "mountains streams ink", "literati painting", "monochrome ink painting",
    "figure painting ink", "album of landscapes", "ink and color on silk",
]

# Chỉ nhận chất liệu mực. Loại đồ gốm/đồng/dệt lọt vào theo từ khoá.
MEDIUM_OK = re.compile(r"\bink\b", re.IGNORECASE)
MEDIUM_REJECT = re.compile(
    r"porcelain|stoneware|bronze|jade|lacquer|textile|embroider|ceramic|"
    r"woodblock print|earthenware", re.IGNORECASE)

# THƯ PHÁP PHẢI BỊ LOẠI. Met xếp thư pháp chung department với hội hoạ, mà một ô
# 512px cắt từ trục cuốn thư pháp thì toàn chữ Hán. Train vào đó, LoRA sẽ học
# "phong cách này = có chữ" rồi rắc chữ lên mọi khung hình — đúng cái mà negative
# prompt "text, watermark, letters" đang phải chống.
TITLE_REJECT = re.compile(
    r"calligraph|couplet|\bpoem|\bpoems\b|song of the|sutra|inscription|"
    r"colophon|letter to|quatrain", re.IGNORECASE)
CLASSIFICATION_REJECT = re.compile(r"calligraph", re.IGNORECASE)

# Chỉ giữ tranh Trung Hoa. Met gộp cả Nhật/Ấn/Triều Tiên vào department Asian Art,
# nhưng ukiyo-e hay tiểu hoạ Ấn Độ là phong cách KHÁC hẳn thủy mặc.
CULTURE_REJECT = re.compile(
    r"japan|korea|india|nepal|tibet|thai|vietnam|persia|islamic", re.IGNORECASE)
TITLE_CULTURE_REJECT = re.compile(
    r"genji|hindu|nanshoku|otsu-e|ōtsu-e|ukiyo|samurai|kabuki", re.IGNORECASE)

TILE = 512
# Ô có độ lệch chuẩn độ sáng dưới ngưỡng này = mảng giấy trống, không có nét vẽ.
MIN_TILE_STD = 12.0

# --- Lọc theo NỘI DUNG ô ảnh -------------------------------------------------
# Lọc metadata không đủ: thư pháp lọt qua khi tiêu đề không có chữ "poem", và
# trục cuốn tranh vẫn kèm nguyên đoạn đề từ + triện đỏ ở rìa. Ba ngưỡng dưới đây
# đo trực tiếp trên pixel nên bắt được cả những ca đó.
MAX_MEAN_SATURATION = 0.32   # thủy mặc gần đơn sắc; cao hơn = tranh màu/thangka
MAX_RED_SEAL_FRAC = 0.045    # tỉ lệ pixel đỏ bão hoà (triện) chiếm khung
MAX_TEXT_COMPONENTS = 55     # số vệt mực nhỏ rời rạc — trang chữ thì rất nhiều
UA = {"User-Agent": "MediaComposer-style-dataset/1.0 (academic use)"}


def _get_json(url: str, timeout: int = 45, retries: int = 4):
    """GET JSON kèm backoff. Met chặn 403 khi gọi quá dày — phải lùi rồi thử lại,
    nếu không cả mẻ sẽ rỗng dù dữ liệu vẫn còn đó."""
    delay = 2.0
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429, 500, 502, 503):
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(delay)
            delay *= 2
    raise last


def search_object_ids(limit_per_query: int) -> list:
    """Gom objectID từ mọi truy vấn, giữ thứ tự và bỏ trùng."""
    seen, ids = set(), []
    for q in QUERIES:
        params = urllib.parse.urlencode({
            "q": q, "departmentId": MET_DEPT_ASIAN_ART, "hasImages": "true"})
        try:
            data = _get_json(f"{MET_SEARCH}?{params}")
        except Exception as e:
            print(f"[WARN] Truy van '{q}' loi: {e}")
            continue
        got = data.get("objectIDs") or []
        print(f"[INFO] '{q}': {data.get('total', 0)} ket qua")
        for oid in got[:limit_per_query]:
            if oid not in seen:
                seen.add(oid)
                ids.append(oid)
    return ids


def is_usable(obj: dict) -> tuple:
    """Trả (dùng được?, lý do loại) để log ra được vì sao dataset còn/mất ảnh."""
    if not obj.get("isPublicDomain"):
        return False, "khong_public_domain"
    image_url = obj.get("primaryImage") or ""
    if not image_url:
        return False, "khong_co_anh"
    # URL Met đôi khi chứa khoảng trắng → urllib từ chối. Bỏ qua, không đáng cứu.
    if any(ch.isspace() for ch in image_url):
        return False, "url_hong"

    medium = obj.get("medium") or ""
    if MEDIUM_REJECT.search(medium):
        return False, "chat_lieu_khong_phai_tranh"
    if not MEDIUM_OK.search(medium):
        return False, "khong_phai_muc"

    title = obj.get("title") or ""
    if TITLE_REJECT.search(title):
        return False, "thu_phap"
    if CLASSIFICATION_REJECT.search(obj.get("classification") or ""):
        return False, "thu_phap"

    culture = " ".join(str(obj.get(k) or "") for k in
                       ("culture", "artistNationality", "country"))
    if CULTURE_REJECT.search(culture) or TITLE_CULTURE_REJECT.search(title):
        return False, "khong_phai_trung_hoa"

    return True, ""


def build_caption(obj: dict, style_token: str) -> str:
    """Caption mô tả NỘI DUNG bức tranh, mở đầu bằng trigger token.

    Caption riêng cho từng ảnh là điều kiện bắt buộc để LoRA tách được phong cách
    khỏi nội dung — xem chú thích trong train_style_lora.py.
    """
    bits = []
    title = (obj.get("title") or "").strip()
    if title and title.lower() not in ("untitled", "unknown"):
        # Bỏ phần chú thích trong ngoặc (số hiệu album, ghi chú lưu trữ)
        title = re.sub(r"\([^)]*\)", "", title).strip(" ,;")
        if title:
            bits.append(title.lower())
    classification = (obj.get("classification") or "").strip().lower()
    if classification and classification not in ("paintings",):
        bits.append(classification)
    medium = (obj.get("medium") or "").split(",")[0].strip().lower()
    if medium:
        bits.append(medium)
    content = ", ".join(bits) or "a traditional chinese painting"
    return f"{style_token}, {content}"


def tile_reject_reason(tile: Image.Image):
    """Soi 1 ô 512x512, trả lý do loại hoặc None nếu dùng được.

    Ba tín hiệu, đều tính trên pixel nên không phụ thuộc nhãn của bảo tàng:
    - độ bão hoà trung bình: thủy mặc gần đơn sắc, tranh màu/thangka thì không;
    - tỉ lệ pixel đỏ bão hoà: bắt ô bị triện son chiếm chỗ;
    - số vệt mực nhỏ rời rạc: trang thư pháp cho ra hàng trăm vệt cỡ đều nhau,
      tranh thì ít mảng và mảng to.
    """
    import cv2
    import numpy as np

    arr = np.asarray(tile.convert("RGB"))
    gray = np.asarray(tile.convert("L"), dtype="float32")

    if gray.std() < MIN_TILE_STD:
        return "o_trong"

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0

    if float(sat.mean()) > MAX_MEAN_SATURATION:
        return "tranh_mau"

    # Triện son: đỏ (hue quanh 0/180 trong thang OpenCV 0-179), bão hoà, đủ sáng
    red = (((hue < 12) | (hue > 168)) & (sat > 0.45) & (val > 0.25))
    if float(red.mean()) > MAX_RED_SEAL_FRAC:
        return "trien_son_chiem_khung"

    # Vệt mực nhỏ rời rạc → chữ. Ngưỡng Otsu đảo để mực thành foreground.
    g8 = np.asarray(tile.convert("L"))
    _, binary = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else []
    small = int(((areas >= 20) & (areas <= 1500)).sum()) if len(areas) else 0
    if small > MAX_TEXT_COMPONENTS:
        return "trang_chu"

    return None


def tiles_from(img: Image.Image) -> list:
    """Cắt trượt dọc cạnh dài thành các ô vuông TILE, bỏ ô gần trắng."""
    import numpy as np

    w, h = img.size
    short = min(w, h)
    if short < TILE:
        scale = TILE / short
        img = img.resize((max(TILE, int(w * scale)), max(TILE, int(h * scale))),
                         Image.LANCZOS)
        w, h = img.size
        short = min(w, h)

    # Chuẩn hoá cạnh ngắn về TILE rồi trượt trên cạnh dài
    scale = TILE / short
    img = img.resize((max(TILE, int(round(w * scale))),
                      max(TILE, int(round(h * scale)))), Image.LANCZOS)
    w, h = img.size

    boxes = []
    if w >= h:
        n = max(1, min(8, w // TILE))
        step = (w - TILE) / max(1, n - 1) if n > 1 else 0
        for i in range(n):
            x = int(round(i * step))
            boxes.append((x, 0, x + TILE, TILE))
    else:
        n = max(1, min(8, h // TILE))
        step = (h - TILE) / max(1, n - 1) if n > 1 else 0
        for i in range(n):
            y = int(round(i * step))
            boxes.append((0, y, TILE, y + TILE))

    out, rejects = [], []
    for box in boxes:
        tile = img.crop(box)
        reason = tile_reject_reason(tile)
        if reason is None:
            out.append(tile)
        else:
            rejects.append(reason)
    return out, rejects


def main():
    p = argparse.ArgumentParser(description="Dung dataset LoRA phong cach tu Met Open Access")
    p.add_argument("--style", default="thuy_mac")
    p.add_argument("--style_token", default="thuymac ink wash style")
    p.add_argument("--target", type=int, default=80, help="So anh 512x512 muon co")
    p.add_argument("--max_tiles_per_work", type=int, default=3,
                   help="Toi da bao nhieu o lay tu MOT buc (giu da dang nguon)")
    p.add_argument("--per_query", type=int, default=40)
    p.add_argument("--delay", type=float, default=0.8,
                   help="Giay nghi giua cac object (Met chan 403 neu goi qua day)")
    p.add_argument("--out_dir", default="")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        _MC_ROOT, "storage", "style_datasets", args.style)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Dich: {args.target} anh {TILE}x{TILE} -> {out_dir}")
    ids = search_object_ids(args.per_query)
    print(f"[INFO] Tong {len(ids)} objectID ung vien\n")

    manifest, n_saved, n_works = [], 0, 0
    rejected = {}
    for oid in ids:
        if n_saved >= args.target:
            break
        try:
            obj = _get_json(f"{MET_OBJECT}/{oid}")
        except Exception as e:
            rejected["loi_api"] = rejected.get("loi_api", 0) + 1
            print(f"[WARN] objectID {oid}: {e}")
            continue
        ok, reason = is_usable(obj)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        url = obj["primaryImage"]
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            print(f"[WARN] Tai anh {oid} loi: {e}")
            continue

        tiles, tile_rejects = tiles_from(img)
        for r in tile_rejects:
            rejected[r] = rejected.get(r, 0) + 1
        tiles = tiles[:args.max_tiles_per_work]
        if not tiles:
            continue
        n_works += 1
        caption = build_caption(obj, args.style_token)

        for k, tile in enumerate(tiles):
            if n_saved >= args.target:
                break
            stem = f"{args.style}_{oid}_{k}"
            tile.save(os.path.join(out_dir, f"{stem}.png"))
            with open(os.path.join(out_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write(caption)
            manifest.append({
                "file": f"{stem}.png",
                "met_object_id": oid,
                "title": obj.get("title", ""),
                "artist": obj.get("artistDisplayName", ""),
                "date": obj.get("objectDate", ""),
                "medium": obj.get("medium", ""),
                "license": "CC0 / Public Domain (Met Open Access)",
                "source_url": obj.get("objectURL") or f"{MET_OBJECT_PAGE}/{oid}",
                "image_url": url,
                "caption": caption,
            })
            n_saved += 1

        print(f"[{n_saved:>3}/{args.target}] {len(tiles)} o <- {obj.get('title','')[:52]} "
              f"({obj.get('objectDate','')})")
        time.sleep(args.delay)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "style": args.style,
            "style_token": args.style_token,
            "n_images": n_saved,
            "n_source_works": n_works,
            "source": "The Metropolitan Museum of Art Open Access (CC0)",
            "items": manifest,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[XONG] {n_saved} anh tu {n_works} buc goc -> {out_dir}")
    if rejected:
        print("[INFO] Da loai:", ", ".join(
            f"{k}={v}" for k, v in sorted(rejected.items(), key=lambda x: -x[1])))
    if n_saved < 40:
        print(f"[CANH BAO] Chi co {n_saved} anh (<40). Tang --per_query hoac "
              f"--max_tiles_per_work, hoac bo sung anh thu cong.")
    print("[INFO] Buoc tiep theo:")
    print(f'       scripts\\train_style_lora.py --style {args.style} '
          f'--images_dir "{out_dir}" --style_token "{args.style_token}" --steps 1500')


if __name__ == "__main__":
    main()
