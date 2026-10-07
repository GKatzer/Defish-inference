"""Draws the boxes the service returns on a photo (the service does not return a picture with class labels:
/analyze returns the original photo, /detect a picture with plain boxes in the detector's own style).

usage: python docs/examples/annotate.py http://127.0.0.1:8000 photo.jpg out.jpg [--max-width 1200] [--classes]
Without --classes the photo is sent to /detect and each box carries the detector score. With --classes it is sent to
/analyze and each box carries the class and its probability (green: confident, orange: flagged `uncertain`); that
needs the classifier, and with stand-in files the classes mean nothing. Boxes are in the coordinates of the original
photo. Needs httpx and opencv-python-headless (both in requirements.txt).
"""
import argparse
import base64
import os

import cv2
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("base_url")
parser.add_argument("photo")
parser.add_argument("out")
parser.add_argument("--max-width", type=int, default=1200)
parser.add_argument("--classes", action="store_true")
args = parser.parse_args()

raw = open(args.photo, "rb").read()
if args.classes:
    answer = httpx.post(args.base_url + "/analyze", json={"image_bytes": base64.b64encode(raw).decode()}, timeout=300)
else:
    answer = httpx.post(args.base_url + "/detect", files={"image": (os.path.basename(args.photo), raw, "image/jpeg")}, timeout=120)
answer.raise_for_status()
detections = answer.json()["detections"]

image = cv2.imread(args.photo)
scale = min(1.0, args.max_width / image.shape[1])
if scale < 1.0:
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
thickness = 2 if image.shape[1] >= 1000 else 1
font = 0.45 if image.shape[1] >= 1000 else 0.4
for d in detections:
    x1, y1, x2, y2 = (int(v * scale) for v in d["bbox"])
    colour = (0, 255, 0)
    if args.classes:
        text = f"{d['class']} {d['class_confidence']:.2f}"
        if d["uncertain"]:
            colour = (0, 165, 255)  # orange (BGR)
    else:
        text = f"{d['confidence']:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
    cv2.putText(image, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thickness, cv2.LINE_AA)
cv2.imwrite(args.out, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"{os.path.basename(args.photo)}: {len(detections)} boxes -> {args.out} ({os.path.getsize(args.out) // 1024} KB)")
