"""Wall-clock time of POST /detect for each photo (client and service on the same machine).

usage: python docs/examples/measure_latency.py http://127.0.0.1:8000 docs/examples/photos/*.jpg [--runs 10]
Start the service with RATE_LIMIT_REQUESTS raised (e.g. 1000): the script sends 11 requests per photo, which exceeds
the default limit of 60 per minute for four photos. /analyze is deliberately not timed here: with the stand-in
classifier its embedder cost is not representative.
"""
import argparse
import os
import statistics
import time

import cv2
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("base_url")
parser.add_argument("photos", nargs="+")
parser.add_argument("--runs", type=int, default=10)
args = parser.parse_args()

print(f"{'photo':36} {'size':>10} {'fish':>5} {'median':>8} {'min':>7} {'max':>7}   (ms, {args.runs} runs after 1 warm-up)")
for path in args.photos:
    raw = open(path, "rb").read()
    h, w = cv2.imread(path).shape[:2]
    times, count = [], None
    for i in range(args.runs + 1):
        started = time.perf_counter()
        r = httpx.post(args.base_url + "/detect", files={"image": (os.path.basename(path), raw, "image/jpeg")}, timeout=120)
        elapsed = (time.perf_counter() - started) * 1000
        r.raise_for_status()
        count = r.json()["count"]
        if i:
            times.append(elapsed)
    print(f"{os.path.basename(path):36} {f'{w}x{h}':>10} {count:>5} {statistics.median(times):8.0f} {min(times):7.0f} {max(times):7.0f}")
