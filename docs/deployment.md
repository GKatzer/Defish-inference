# Deployment

How the service is meant to run, what it needs on disk, and what was and was not verified. No real addresses appear here; `<private-address>` stands for whatever interface the container's port is published on.

- [What has to be on the machine](#what-has-to-be-on-the-machine)
- [Without Docker](#without-docker)
- [With Docker Compose](#with-docker-compose)
- [Running it in production](#running-it-in-production)
- [Updating the models](#updating-the-models)
- [Upgrading from the previous version of this repository](#upgrading-from-the-previous-version-of-this-repository)
- [What was verified](#what-was-verified)

## What has to be on the machine

| Need | Detail |
|---|---|
| Python 3.12 | the Dockerfile uses `python:3.12-slim`; the tests and the examples were run on 3.12.3 |
| Python packages | `requirements.txt` (pinned: fastapi 0.121.2, uvicorn 0.38.0, onnxruntime 1.23.2, opencv-python-headless 4.12.0.88, pydantic-settings 2.12.0, numpy 1.26 or newer); `requirements-dev.txt` adds pytest and onnx |
| System libraries for OpenCV | `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `libgomp1` (installed by the Dockerfile; the headless wheel needs fewer on many distributions) |
| Model files | under `./models` relative to the working directory: `best.onnx` and `disease_classifier/` with `meta.json`, `head_numpy.npz` and the embedder named in `meta.json` ([contract](architecture.md#model-files-and-their-contract)). None of them is in the repository (the classifier files weigh 348 MB) |
| A `.env` | one required line, `vds1_ip=...`, which the settings model demands and nothing uses; template: [env-example.txt](../env-example.txt) |
| CPU | inference runs on `CPUExecutionProvider` only; there is no GPU switch (the detector file has a comment about it). Memory use was not measured |

## Without Docker

```bash
uv venv && uv pip install -r requirements.txt
cp env-example.txt .env
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

Start it from the repository root, because the model paths are relative. `python main.py` does not work ([known issues](design-decisions.md#known-issues-not-changed)). The log goes to standard error at INFO level, with the results of every request (boxes and classes, not the image); nothing is written to files.

## With Docker Compose

```bash
cp env-example.txt .env          # set TAILSCALE_IP and ML_PORT; the defaults bind 127.0.0.1:8000
docker compose up -d --build
docker compose logs -f
curl -s http://127.0.0.1:8000/health
```

What `docker-compose.yml` does:

| Item | Value |
|---|---|
| service / container | `ml-service` / `ml-inference-service`, `restart: unless-stopped` |
| port | the container's 8000 is published on `${TAILSCALE_IP}:${ML_PORT}` (both variables are required; the name comes from the deployment this was written for, where the port is published on a private-network address only) |
| volumes | the whole checkout over `/app`, plus `./models`, `./temp`, `./logs` (the code writes nothing to `temp` or `logs`; the `./models` mount makes the model files come from the host) |
| environment | `.env` as `env_file`; additionally `PYTHONUNBUFFERED=1`, `TZ=Europe/Moscow` and `VDS1_URL=http://${vds1_ip}:${vds1_port}` (no code reads it) |
| network | one bridge network, `ml-network` |
| declared but unused | a named volume `redis_data` |
| command | the Dockerfile's `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` |

A `.dockerignore` keeps `.env`, `.venv`, `models/`, `tests/`, `docs/`, `.git` and caches out of the image (before it existed, `COPY . .` took the whole build directory in, secrets and weights included: 1.64 GB with the model files in the context; now 878 MB, with no `.env` and an empty `models/` in the image, checked by running it without the compose mounts). The model files therefore have to be mounted, which the compose file does (`./models:/app/models`); with plain `docker run`, add `-v "$PWD/models:/app/models"`.

The `--reload` flag together with the mounted checkout is a development setup: the server restarts when a file under `/app` changes, and a restart reloads the models (the 348 MB embedder takes a moment). For production, replace the command (for example with `uvicorn main:app --host 0.0.0.0 --port 8000` in the compose file) and drop the `.:/app` mount.

## Running it in production

- **Network.** The service has no authentication and CORS is open. Keep it on a private network and let one trusted component (the API in front of it) call it; do not publish it to the internet.
- **Rate limit.** The default admits 60 requests per minute per client address, in the memory of one process. A single backend that calls on behalf of all its users shares that allowance (the analysis itself is the tighter limit on one CPU process); raise `RATE_LIMIT_REQUESTS` (and `RATE_LIMIT_WINDOW` if wanted) if it is not enough; with several processes each one counts separately.
- **Health.** Poll `GET /health` and read the body: the HTTP status is 200 even when `"status"` is `"degraded"`.
- **Timeouts.** The service gives up after 300 s (HTTP 408, constant in `api/endpoints.py`); a caller should wait at least as long. `Defish-backend` currently waits 180 s, so on a slow host it gives up first: raise its timeout to match.
- **Load.** `/detect` blocks the event loop while it runs ([measurement](examples/transcripts/health-under-load.txt)); use `/analyze` (which uses a thread pool) unless detection alone is what is needed, or run more processes.
- **Duration.** With the real classifier `/analyze` takes about 0.28 s plus 0.2 to 0.3 s per fish on the test machine (3.2 s for 11 fish, 5.4 s for 29, 12.5 s for 46); the 300 s timeout is far away on that machine, but a caller's own timeout should allow for photos with many fish. A 2-core Xeon VDS needed 69.5 s for 29 fish (2.4 s per fish; 35.0 s with `CLASSIFIER_TTA=0`, see [Configuration](../README.md#configuration)), and 173 s while another container was using its CPU.
- **Photos.** A request body is read whole and decoded fully; size limits belong in the proxy.

## Updating the models

The models are loaded once, at start-up. Replace the files and restart the service. The detector file is a drop-in replacement as long as its input is `[batch, 3, height, width]` and its output `[batch, 5, anchors]` for one class (`Defish-ML-train` documents the export). For the classifier, the three pieces (embedder, `head_numpy.npz`, `meta.json`) come from one training run and must be replaced together: the head's columns are the embedder's outputs, and `meta.json` holds the class order and the gate threshold.

## Upgrading from the previous version of this repository

- `GET /health` can return `"status": "degraded"` (still HTTP 200); it returned `"healthy"` regardless of the models before.
- A missing classifier no longer takes the detector down; `/detect` works without it.
- `/detect` and `/analyze` answer 400 (instead of 500) for input that is not an image or not valid, and 500 bodies are the short `Detection failed` instead of the exception text.
- The picture in `/detect`'s `result_image_b64` now has the photo's colours (red and blue were swapped), and `/detect` no longer writes `/tmp/api_detection_result.jpg`.
- New optional settings `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`; the default rises from 5 to 60 requests per 60 s (at the owner's request: the old value capped the whole demo behind `Defish-backend` at about five analyses a minute).
- **Updating by `git pull`:** the revision that stops tracking `models/best.onnx` (and its `.bak` copy) deletes those files from a working tree that tracked them when it is pulled. Copy `models/` somewhere safe first, or pull into a clone that does not serve traffic. The same revision deletes `CI_CD/cron_auto_push.sh` and the old scripts.

## What was verified

Run on 2026-10-04 on one Linux machine (Python 3.12.3, uv 0.12.18, Docker 29.8.1) in a clean copy of what a commit contains:

| Check | Result |
|---|---|
| `uv venv`, `uv pip install -r requirements-dev.txt`, `pytest` | 69 passed ([transcript](examples/transcripts/tests.txt)) |
| the service started with `uvicorn main:app` from the repository root with the **real detector and classifier files** (2026-10-05) | answers as in [the walkthrough](examples/transcripts/walkthrough.txt); `/analyze` on four photos: [summary](examples/transcripts/analyze-summary.txt); timings and comparisons with the reference runtime: [measurements](examples/transcripts/classifier-measurements.txt) |
| the same in a tree without any weights, with both stand-ins written by `make_standin_models.py ... --detector` (README step 2) | `/health` healthy; `/detect` returns the three fixed boxes; `/analyze` returns three entries |
| `docker compose config -q` with the template `.env` | valid; port bound to `127.0.0.1:8000` |
| `docker compose up -d --build` in a copy of the repository's contents with the real model files and a template `.env` (2026-10-05) | run 1 (no `.dockerignore`): from nothing in 1 min 35 s (base image pulled, packages installed), image 1.64 GB; run 2 (with it): 6 s with the layer cache, image 878 MB, no `.env` and no models in the image; both times the container was healthy and `/analyze` through it gave the same classes, confidences and flags as the native run; torn down with `docker compose down --rmi local` ([transcript](examples/transcripts/docker-compose.txt)) |
| memory use, GPU, Python 3.11, the CPU of the real deployment | **not measured** |
