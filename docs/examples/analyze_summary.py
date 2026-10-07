"""What POST /analyze says about each photo: crops, time, how many fish per class, how many are flagged uncertain.

usage: python docs/examples/analyze_summary.py http://127.0.0.1:8000 docs/examples/photos/*.jpg
Start the service with RATE_LIMIT_REQUESTS raised if you have many photos. Needs httpx (requirements.txt).
"""
import base64
import collections
import os
import sys
import time

import httpx

base, photos = sys.argv[1], sys.argv[2:]
print(f"{'photo':34} {'fish':>5} {'ms':>6} {'uncertain':>10}   classes (count, how many of them uncertain)")
for path in photos:
    raw = open(path, "rb").read()
    started = time.perf_counter()
    answer = httpx.post(base + "/analyze", json={"image_bytes": base64.b64encode(raw).decode()}, timeout=300)
    ms = (time.perf_counter() - started) * 1000
    answer.raise_for_status()
    found = answer.json()["detections"]
    per_class = collections.Counter(d["class"] for d in found)
    unsure = collections.Counter(d["class"] for d in found if d["uncertain"])
    classes = ", ".join(f"{name} {n} ({unsure[name]})" for name, n in per_class.most_common())
    print(f"{os.path.basename(path):34} {len(found):5d} {ms:6.0f} {sum(unsure.values()):10d}   {classes}")
