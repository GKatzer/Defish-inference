# API reference

Plain HTTP and JSON, no authentication, CORS open to every origin. Interactive OpenAPI pages are served by FastAPI at `/docs` and `/openapi.json`. The examples are real answers of the service recorded on 2026-10-05 with the **real detector and classifier** (files that are not in the repository; the true condition of the fish in the example photo is not known); see [examples/transcripts/walkthrough.txt](examples/transcripts/walkthrough.txt), which `docs/examples/walkthrough.sh` reproduces. Floats are rounded and base64 strings cut for reading.

| Route | Purpose | Rate limited |
|---|---|---|
| `GET /` | name, links, whether the models are loaded | no |
| `GET /health` | `healthy` or `degraded`, per-model state | no |
| `GET /model-status` | per-model `loaded` flag and class name | no |
| `GET /test` | constant answer, for a smoke test | no |
| `POST /detect` | fish detection only; multipart upload | yes |
| `POST /analyze` | detection plus classification of every fish; JSON with base64 | yes |

Rate limit: by default 60 requests per 60 s per client address (one a second on average), counted for `/detect` and `/analyze` (also when the request is then rejected as invalid); settings `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW` ([configuration](../README.md#configuration)). The answer to the first request over the limit is `429`:

```json
{"detail": {"error": "Too many requests", "retry_after": 59, "limit": 5, "window": 60}}
```

There is no `Retry-After` header; the seconds are in the body.

## `GET /health`

```console
$ curl -s http://127.0.0.1:8000/health
{"status": "healthy", "service": "ml-inference", "version": "1.0.0", "detector": "loaded", "classifier": "loaded"}
```

`status` is `"healthy"` only when both models are loaded, otherwise `"degraded"`; the HTTP status is 200 either way. `detector` and `classifier` are `"loaded"` or `"failed"`. With the classifier files missing the answer is `{"status": "degraded", ..., "detector": "loaded", "classifier": "failed"}` ([recorded](examples/transcripts/fixes-before-after.txt)).

## `GET /model-status`

```json
{"detector": {"loaded": true, "type": "YOLO_ONNX_Inference"}, "classifier": {"loaded": true, "type": "DinoV2Classifier"}}
```

`loaded` is false and `type` null for a model that failed to load. The keys exist only after start-up has run.

## `GET /` and `GET /test`

`/` returns `{"message": "ML Inference Service", "docs": "/docs", "health": "/health", "detector": "loaded" | "not loaded", "classifier": "loaded" | "failed"}`. `/test` returns `{"message": "API is working", "status": "ok"}`.

## `POST /detect`

Multipart form with one file field, `image`; its content type must start with `image/`.

```console
$ curl -s -F image=@docs/examples/photos/planted-tank-tetras.jpg http://127.0.0.1:8000/detect
{
  "status": "success",
  "count": 11,
  "detections": [
    {"bbox": [848.647, 695.816, 969.477, 774.979], "confidence": 0.896, "class_id": 0},
    {"bbox": [716.384, 631.518, 804.316, 670.279], "confidence": 0.832, "class_id": 0},
    "... 9 more"
  ],
  "result_image_b64": "/9j/4AAQSkZJRgABAQAAAQAB...",
  "message": "Found 11 objects"
}
```

| Field | Meaning |
|---|---|
| `status` | always `"success"` on a 200 |
| `count` | number of detections |
| `detections[].bbox` | `[x1, y1, x2, y2]` in pixels of the decoded photo (EXIF orientation applied), floats, clipped to the photo |
| `detections[].confidence` | detector score, above 0.25; the list is in descending score order (non-maximum suppression at IoU 0.45) |
| `detections[].class_id` | always `0` (one class, fish) |
| `result_image_b64` | the photo as a base64 JPEG (no `data:` prefix) with the boxes drawn as 1-pixel green rectangles labelled `0: <score>`; `null` if encoding failed |
| `message` | `Found N objects` |

A box that lies in the letterbox padding is clipped to a line (`y1 == y2` or `x1 == x2`) and is still returned here; see [architecture.md](architecture.md#coordinate-frames).

Errors: `400` `{"detail": "Uploaded file must be an image"}` (content type), `400` `{"detail": "Cannot decode image"}` (bytes that are not an image, an empty file included), `422` (no `image` field; FastAPI's validation body), `429`, `500` `{"detail": "Detector model not loaded"}` or `{"detail": "Detection failed"}`, `408` after 180 s.

## `POST /analyze`

JSON body with one field read by the service, `image_bytes`: the **base64 of the image file** (any format that OpenCV can decode; the examples use JPEG). Other fields are ignored; `Defish-backend` also sends `filename`, `content_type` and `user_id`.

```console
$ printf '{"image_bytes":"%s"}' "$(base64 -w0 docs/examples/photos/planted-tank-tetras.jpg)" \
    | curl -s -H "Content-Type: application/json" -d @- http://127.0.0.1:8000/analyze
{
  "detections": [
    {
      "bbox": [848.647, 695.816, 969.477, 774.979],
      "det_confidence": 0.896,
      "class": "healthy",
      "class_confidence": 0.999,
      "uncertain": false,
      "top3": [
        {"label": "healthy", "confidence": 0.999},
        {"label": "fin_rot", "confidence": 0.0},
        {"label": "oodiniosis", "confidence": 0.0}
      ]
    },
    "... 10 more"
  ],
  "image": "/9j/4AAQSkZJRgABAQAAAQAB...",
  "image_width": 1600,
  "image_height": 1229
}
```

On this photo 3 of the 11 fish are flagged `uncertain`; the first one is not (0.999 is above the 0.83 gate).

| Field | Meaning |
|---|---|
| `detections[].bbox` | as in `/detect`, in the coordinates of `image` |
| `detections[].det_confidence` | the detector's score |
| `detections[].class` | one of `healthy`, `fin_rot`, `dermatomycosis`, `hexamitosis`, `mycobacteriosis`, `oodiniosis`, `plistophorosis` (the `classes` of `meta.json`) |
| `detections[].class_confidence` | probability of `class` from the softmax of the linear head; not verified to be calibrated |
| `detections[].uncertain` | `true` when `class_confidence` is below the threshold of `meta.json` (0.83 in the shipped file); the label is still given |
| `detections[].top3` | the three most probable classes with their probabilities, the first equal to `class` |
| `image` | the decoded photo re-encoded as a base64 JPEG (OpenCV's default quality, 95); upright, because the EXIF orientation was applied while decoding |
| `image_width`, `image_height` | size of that image in pixels |

Every box the detector produced that can be cropped has an entry; weak classifications are flagged, not removed. A box that cannot be cropped (zero area after clipping) is skipped, so `detections` can be shorter than the `/detect` list for the same photo. A photo with no fish gives `"detections": []` and a normal 200.

Errors: `400` `{"detail": "Body must be JSON with a base64 string in 'image_bytes'"}` (not JSON, no field, not a string, invalid base64), `400` `{"detail": "Cannot decode image"}`, `429`, `500` `{"detail": "Detector model not loaded"}`, `{"detail": "Classifier model not loaded"}` or `{"detail": "Detection failed"}`, `408` `{"detail": "Processing timeout. Image might be too large."}` after 180 s.

Time: about 0.28 s for the detector plus 0.2 to 0.3 s per fish with the real classifier (3.2 s for 11 fish, 5.4 s for 29, 12.5 s for 46, one process, [measured](examples/transcripts/analyze-summary.txt)); the service gives up after 180 s.

Request size: the body is read whole; there is no limit in the service (put one in the reverse proxy). The base64 text is a third larger than the file.

## What `Defish-backend` sends and reads

Its worker POSTs `{"image_bytes", "filename", "content_type", "user_id"}` to `/analyze` with a 180 s timeout, treats any status other than 200 as a failed analysis, and reads `bbox`, `det_confidence`, `class`, `class_confidence`, `uncertain`, `top3`, `image`, `image_width` and `image_height` from the answer (`Defish-backend/worker.py`, read, not run against this service). Its `GET /handshake` calls `/health` and returns the text unchanged.
