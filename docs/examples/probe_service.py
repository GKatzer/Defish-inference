"""Black-box checks of a running service, printed as a transcript (used for docs/examples/transcripts/).

usage:
    python docs/examples/probe_service.py http://127.0.0.1:8000 health
    python docs/examples/probe_service.py http://127.0.0.1:8000 bad-input colours --photo docs/examples/photos/cardinal-tetra-school.jpg
    python docs/examples/probe_service.py http://127.0.0.1:8000 limiter

Groups: health (2 requests), bad-input (4), colours (1, needs a writable /tmp on the same host for the tmp check),
photo (2, needs --photo), limiter (LIMIT + 2 requests with an empty body; the last two must be refused; LIMIT is
--limit, default 60, the default of the service).
The service counts every request toward its per-IP limit, even a failing one, so against a service with a lower limit
(for example the original code, where it was 5 and fixed) pass --limit with that number.
Needs httpx and numpy (both in requirements.txt) and opencv-python-headless.
"""
import argparse
import base64
import itertools
import json
import os
import time

import cv2
import httpx
import numpy as np

TMP_RESULT = "/tmp/api_detection_result.jpg"  # written by the original /detect implementation


def show(title, response, limit=300):
    body = response.text
    if len(body) > limit:
        body = body[:limit] + f"... ({len(response.text)} characters)"
    print(f"{title}\n  HTTP {response.status_code}  {body}")


def group_health(base, args):
    show("GET /health", httpx.get(base + "/health"))
    show("GET /model-status", httpx.get(base + "/model-status"))


def group_bad_input(base, args):
    show("POST /analyze  body {}", httpx.post(base + "/analyze", json={}))
    show("POST /analyze  body 'not json'", httpx.post(base + "/analyze", content=b"not json", headers={"Content-Type": "application/json"}))
    payload = {"image_bytes": base64.b64encode(b"this is not an image").decode()}
    show("POST /analyze  image_bytes = base64 of text", httpx.post(base + "/analyze", json=payload))
    show("POST /detect   file of type image/jpeg containing text",
         httpx.post(base + "/detect", files={"image": ("x.jpg", b"this is not an image", "image/jpeg")}))


def group_colours(base, args):
    """A solid red 200x100 picture goes through /detect (no fish found, so the answer is the picture itself)."""
    patch = np.zeros((100, 200, 3), np.uint8)
    patch[:] = (0, 0, 255)  # BGR red
    ok, jpg = cv2.imencode(".jpg", patch)
    started = time.time()
    response = httpx.post(base + "/detect", files={"image": ("red.jpg", jpg.tobytes(), "image/jpeg")})
    body = response.json()
    print(f"POST /detect  solid red picture\n  HTTP {response.status_code}  count={body.get('count')}  keys={sorted(body)}")
    image_b64 = body.get("result_image_b64")
    if image_b64:
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
        b, g, r = (int(v) for v in decoded[50, 100])
        print(f"  centre pixel of result_image_b64 decoded as BGR: ({b}, {g}, {r})  ->  {'red, as sent' if r > 200 and b < 50 else 'colours swapped' if b > 200 and r < 50 else 'other'}")
    if os.path.exists(TMP_RESULT):
        print(f"  {TMP_RESULT} exists, modified {time.time() - os.path.getmtime(TMP_RESULT):.0f} s ago (request started {time.time() - started:.0f} s ago)")
    else:
        print(f"  {TMP_RESULT} does not exist")


def group_photo(base, args):
    with open(args.photo, "rb") as f:
        raw = f.read()
    name = os.path.basename(args.photo)
    response = httpx.post(base + "/detect", files={"image": (name, raw, "image/jpeg")}, timeout=120)
    body = response.json()
    print(f"POST /detect  {name}\n  HTTP {response.status_code}  count={body.get('count')}  first={body['detections'][:1]}")
    response = httpx.post(base + "/analyze", json={"image_bytes": base64.b64encode(raw).decode()}, timeout=120)
    body = response.json()
    detections = body.get("detections", [])
    print(f"POST /analyze {name}\n  HTTP {response.status_code}  detections={len(detections)}  size={body.get('image_width')}x{body.get('image_height')}")
    for d in detections[:2]:
        print("   ", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != 'top3'}))


def group_limiter(base, args):
    count = args.limit + 2
    responses = [httpx.post(base + "/analyze", json={}) for _ in range(count)]
    runs = [(code, len(list(group))) for code, group in itertools.groupby(r.status_code for r in responses)]
    print(f"{count} x POST /analyze  body {{}}\n  statuses: {', then '.join(f'{n} x {code}' for code, n in runs)}")
    refused = next((r for r in responses if r.status_code == 429), None)
    if refused is not None:
        print(f"  first refusal: {refused.text}")


GROUPS = {"health": group_health, "bad-input": group_bad_input, "colours": group_colours, "photo": group_photo, "limiter": group_limiter}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_url")
    parser.add_argument("groups", nargs="+", choices=sorted(GROUPS))
    parser.add_argument("--photo")
    parser.add_argument("--limit", type=int, default=60, help="the service's limit per window, for the limiter group")
    args = parser.parse_args()
    for g in args.groups:
        print(f"--- {g}")
        GROUPS[g](args.base_url.rstrip("/"), args)
