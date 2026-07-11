# -*- coding: utf-8 -*-
"""
Detect mặt anime trong 1 ảnh — chạy như TIẾN TRÌNH CON riêng biệt.

Lý do tồn tại: OpenCV (detectMultiScale) và PyTorch nạp chung OpenMP runtime trong
cùng process trên Windows gây crash native (process bị kill không traceback).
Script này CHỈ import cv2/numpy — không torch — nên miễn nhiễm hoàn toàn.

Usage: python detect_faces_cli.py <image_path> <cascade_path>
Output (stdout): JSON list các bbox [x, y, w, h], lớn nhất trước.
Exit code: 0 = OK (kể cả 0 mặt), khác 0 = lỗi.
"""
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> int:
    if len(sys.argv) < 3:
        print("[]")
        return 1

    image_path, cascade_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(image_path) or not os.path.exists(cascade_path):
        print("[]")
        return 1

    import cv2
    cv2.setNumThreads(1)

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("[]")
        return 2

    img = cv2.imread(image_path)
    if img is None:
        print("[]")
        return 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(28, 28))

    if faces is None or len(faces) == 0:
        print("[]")
        return 0

    boxes = sorted(
        [[int(v) for v in f] for f in faces],
        key=lambda f: f[2] * f[3],
        reverse=True,
    )
    print(json.dumps(boxes))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print("[]")
        print(f"CRITICAL_CRASH: {traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
