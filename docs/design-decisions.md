# Design decisions

The decisions that shape this service, what was chosen over what, and what is known to be unfinished. Measurements of the models themselves live in `Defish-ML-train` (data, training, leak-free evaluation); this page quotes them only where a decision rests on them, with the sample sizes. Pipeline details: [architecture.md](architecture.md).

- [1. Detect on a fixed canvas, classify from the original](#1-detect-on-a-fixed-canvas-classify-from-the-original)
- [2. Two different paddings](#2-two-different-paddings)
- [3. A frozen embedder and a linear head](#3-a-frozen-embedder-and-a-linear-head)
- [4. A confidence gate instead of a forced label](#4-a-confidence-gate-instead-of-a-forced-label)
- [5. Calling the embedder once per crop](#5-calling-the-embedder-once-per-crop)
- [6. ONNX Runtime and NumPy only, and the same preprocessing as training](#6-onnx-runtime-and-numpy-only-and-the-same-preprocessing-as-training)
- [7. Two independent models, a health that tells the truth](#7-two-independent-models-a-health-that-tells-the-truth)
- [8. A small in-memory rate limit and open CORS](#8-a-small-in-memory-rate-limit-and-open-cors)
- [9. Stand-in models for tests and examples](#9-stand-in-models-for-tests-and-examples)
- [Considered and not adopted](#considered-and-not-adopted)
- [Changes made with this documentation round](#changes-made-with-this-documentation-round)
- [Known issues, not changed](#known-issues-not-changed)

## 1. Detect on a fixed canvas, classify from the original

**Chosen.** The photo is letterboxed to 960×960 for the detector only; the boxes are mapped back, and each crop for the classifier is cut from the **original** pixels.
**Why.** A fixed input size keeps the detector's cost constant whatever the upload (the commit that introduced it is titled "image to 960x960 letterbox resize logic for faster detection"), padding keeps the proportions undistorted, and cutting the crops from the original keeps the detail that is lost in the downscaled canvas, which matters most for small fish.
**Instead of.** Larger inputs: `Defish-ML-train` trained and ran the detector at 1280 px and found no gain (AP50 0.65) at about 1.8 times the cost. Tiled inference raised recall on tiny fish from 0.43 to 0.60 but doubled false positives per image (0.5 to 1.1) and left AP50 unchanged.
**Cost.** Fish under about 64 px remain the weak spot: recall about 0.3 to 0.45 on the 165-image test split (only 30 fish below 24 px, so the figure is noisy).

## 2. Two different paddings

The detector's canvas is padded **white** (`letterbox_resize`); the classifier's square crops are padded with the neutral colour **(124, 116, 104)**. Each follows the preprocessing under which that model was evaluated or trained (`Defish-ML-train` evaluates the detector through this very letterbox, and its crop preparation uses the same neutral constant), so neither can be "harmonised" without re-evaluating. The detector class also letterboxes its input once more with grey 114 padding; on the already-square canvas this is an exact no-op, so the second padding colour never appears.

## 3. A frozen embedder and a linear head

**Chosen.** Frozen DINOv2-base embeddings (crop plus its mirror image, averaged) feeding a logistic regression with a standard scaler, trained on 138 expert-checked crops of seven classes (`healthy`, `fin_rot`, `dermatomycosis`, `hexamitosis`, `mycobacteriosis`, `oodiniosis`, `plistophorosis`) plus 5 external "silver" healthy crops.
**Why.** 138 crops are too few to train a network from scratch (the earlier classifier trained that way scores lower, table below); a linear head on a strong general embedding has few parameters to overfit and can be evaluated with grouped cross-validation (no fish with crops on both sides of a split).
**Evidence** (`Defish-ML-train`, not reproduced here: the data are not public):

| Comparison | Result | Reading |
|---|---|---|
| grouped cross-validation over the 138 crops | accuracy 0.57 ± 0.04, top-3 0.84; 95% interval about ± 8 points | the number the service ships with (`meta.json`: 0.572 ± 0.036, 0.841) |
| earlier classifier (YOLO classifier trained from scratch) against DINOv2 + head, matched split of 45 unique crops | 0.38 (95% interval 0.25 to 0.52) against 0.51 (0.37 to 0.65) | the intervals overlap: the direction is clear, the size of the difference is not established |
| BioCLIP embeddings instead of DINOv2 | 0.42 | not adopted |
| SAM-masked crops | 0.45 (recall of `fin_rot` 0.56 to 0.35) | the masks clip thin fins, which are the signal for fin rot; not adopted |

**Known weakness.** All six `plistophorosis` crops were neon tetras and the healthy class held none, so "is a neon tetra" and "has the disease" were perfectly correlated. Five external healthy-tetra photos reduced the mislabelled healthy tetras from 3 of 5 to 1 of 5 in a leave-one-out check: better, not solved. The other diseases were not checked for the same failure.

## 4. A confidence gate instead of a forced label

**Chosen.** A prediction whose probability is below the threshold in `meta.json` (0.83) is returned with `"uncertain": true`, together with its `top3`; it is not dropped.
**Why.** With 0.57 accuracy, a plain arg max would present coin-flip answers as findings. The gate trades coverage for trust, and the caller decides how to show a weak guess ("unclear, consult a specialist"). The earlier design had two extra classes (`many_fish`, `not_a_fish`) that swallowed bad crops; they were dropped because they took training signal from the seven real classes and left the caller no fallback.
**Trade-off** (out-of-fold predictions, `Defish-ML-train`):

| threshold | crops kept | accuracy among kept |
|---|---|---|
| 0.00 | 100% | 0.57 |
| 0.45 | 92% | 0.60 |
| 0.65 | 70% | 0.67 |
| 0.83 (shipped) | 49% | 0.75 |

The threshold was picked on the same out-of-fold predictions that produce these figures, so 0.75 is a target the threshold was tuned to reach, not an independently measured accuracy. It is stored in `meta.json`, not in the code, so it can change with a retrained model without a release.
**Behaviour in the service.** The comparison is `confidence < threshold` (a prediction exactly at the threshold is kept, [test](../tests/test_classifier.py)); every detector box that can be cropped gets a result.

## 5. Calling the embedder once per crop

**Before (the original author's choice).** All crops of a photo, each with its mirror image, went through the embedder in one ONNX Runtime call (a `(2N, 3, 224, 224)` tensor; the exported graph has a dynamic batch dimension). The code comment said that one call per crop was the actual bottleneck on photos with many fish, because every call pays the session overhead again.
**Measured with the real embedder, which contradicted it.** On an AMD Ryzen 7 7435HS (16 threads, ONNX Runtime 1.23.2) one call per crop (the crop and its mirror image as a batch of 2) was **1.3 to 1.9 times faster** than one call for all crops in every run (three photos with fish; ONNX Runtime's default threads, 8 and 4 threads; for the 46 fish of the dense-shoal photo 12.1 s against 18.4 s). The cost per image grows with the batch size:

| images per call | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ms per image | 131 | 131 | 142 | 156 | 169 | 191 |

(`measure_classifier.py --scan`, [raw output](examples/transcripts/classifier-measurements.txt); timings vary by up to 30 % between runs.) The answers do not depend on the way of calling: same labels, confidences within 2e-7.
**Now.** `DinoV2Classifier._embed_batch` calls the embedder once per crop; a test checks that there is one call per crop and that its two items are the crop and its mirror image. `/analyze` took 4.3, 11.5 and 18.8 s for the 11, 29 and 46 fish of the example photos before the change and 3.2, 5.4 and 12.5 s after (one request each; [before](examples/transcripts/analyze-summary-before-classifier-changes.txt), [after](examples/transcripts/analyze-summary.txt)). The comment may have been right for the deployment CPU, which was not measured; if it was, a small chunk of crops per call is the thing to try. ONNX Runtime's thread count also matters (one probe run: 186 ms per crop with 8 threads, 257 ms with the default 16); not pursued.
**Open levers** (from `Defish-ML-train`, not done): dropping the mirror pair would halve the cost at an unmeasured accuracy cost; fp16 or int8 quantisation would shrink the 348 MB embedder.

## 6. ONNX Runtime and NumPy only, and the same preprocessing as training

The embedder is exported to ONNX (the image normalisation and the pooling live inside the graph) and the scaler and the linear head are plain arrays in an `.npz`, so the serving environment needs neither PyTorch nor scikit-learn. `Defish-ML-train` checked the export against the PyTorch model (cosine similarity above 0.999 on real crops).
**Cost.** The retraining scripts must be re-run together with the two export scripts after every retraining, and the 348 MB embedder file has to be delivered to the server separately from the code.
**Found with the real files: the crop resize differed from training.** The service resized crops with OpenCV's `INTER_CUBIC`; the training script (`Defish-ML-train/experiments/classifier/train_final.py`) and the reference runtime (`infer_onnx.py`) use Pillow's `BICUBIC`. Compared with that reference on the 86 crops found in the three example photos that contain fish:

| | labels equal | largest difference in the confidence of the label (per photo) | `uncertain` flag differs |
|---|---|---|---|
| before: OpenCV resize (`measure_classifier.py --opencv-resize`) | 81 of 86 (all five differences on the dense-shoal photo) | 0.114, 0.120, 0.053 | 4 of 86 |
| now: Pillow resize | 86 of 86 | 1.0e-7, 1.1e-7, 2.7e-8 | 0 |

So the resize was the whole difference, and the service now reproduces the reference to float precision. No ground truth exists for these photos, so the effect on accuracy was not measured; but the confidence gate (0.83) was tuned on the out-of-fold probabilities of the Pillow pipeline, which makes it the consistent choice. The mechanism was not investigated (the two cubic kernels use different constants, and Pillow also filters over the source when shrinking). **Changed** with the owner's approval: `DinoV2Classifier._preprocess` resizes with Pillow (already in `requirements.txt`, unused until then); a test compares it with a direct Pillow computation and shows that OpenCV's result differs.

## 7. Two independent models, a health that tells the truth

Each model is loaded in its own `try`; a missing file disables that model only, and `/health` reports `"degraded"` and which one. A deployment with only the detector still serves `/detect`. The HTTP status of `/health` stays 200: its only known consumer, `Defish-backend`, passes the body through, and a non-200 would turn a partial outage into a total one for any caller that only checks the status. Demonstrated in [fixes-before-after.txt](examples/transcripts/fixes-before-after.txt).

## 8. A small in-memory rate limit and open CORS

A sliding window per client address in the process memory is the smallest thing that protects a CPU-bound service from one noisy client; it needs no shared store. Its limits: it is per process (several workers multiply the allowance), it keys on the peer address (see [architecture.md](architecture.md#concurrency-timeouts-limits)), and the previous default of 5 requests per minute, fixed in code, suited a single human client: behind `Defish-backend`, which is one client for all its users, it capped the whole demo at about five analyses a minute. The limit is now configurable (`RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`) and the default is 60 per minute, one a second on average; it is still a flood guard, not a capacity plan, because one CPU process finishes far fewer analyses. CORS allows every origin, which the code comment already flags as a placeholder for production; the compose file publishes the port on one chosen address, which is how a private deployment is meant to be run.

## 9. Stand-in models for tests and examples

The real classifier files are not public and weigh 348 MB. [examples/make_standin_models.py](examples/make_standin_models.py) builds files with the same names, graph interface and array names (an "embedder" that is a projection of the mean colour, a fixed random head), and a constant-output detector for the tests. They exercise everything that is code (loading, geometry, batching, the gate, error handling, limits) and prove nothing about accuracy: a stand-in class depends on the crop's average colour only. Every example in this repository that uses them says so; the recorded examples of 2026-10-05 (walkthrough, summaries, pictures) come from the **real** detector and classifier files, which are not in the repository.

## Considered and not adopted

Facts and numbers are in `Defish-ML-train`; listed here because they bear on the serving design.

| Idea | Outcome |
|---|---|
| detector at 1280 px | no gain, about 1.8× cost |
| tiled inference | recall on tiny fish 0.43 → 0.60, false positives per image 0.5 → 1.1, AP50 unchanged |
| self-training the detector on 1781 unlabelled photos | only 199 pseudo-labels passed the two-model agreement filter, all easy large fish; not trained |
| BioCLIP or SAM-masked crops for the classifier | worse (0.42 and 0.45 against 0.57) |
| fp16 / int8 embedder, no mirror pair | not done; levers, not measured |

## Changes made with this documentation round

The documentation was written against the code as it was found; running that code showed four defects (rows 1, 2, 3 and 5) that made the documented behaviour either untrue or hard to state, and two gaps (rows 4 and 6); running it with the real model files and in Docker showed three more (rows 7 to 9). With the owner's approval they were fixed minimally; every fix has a test that fails without it (twelve deliberate regressions were put back by hand in a scratch copy and each was caught), and the before/after output is in [fixes-before-after.txt](examples/transcripts/fixes-before-after.txt).

| # | Found in the original code | Fix |
|---|---|---|
| 1 | with the classifier files absent the **detector** also stayed unloaded (one shared import), and `/health` answered `"status": "healthy"` with both models `"failed"`; the log line for the classifier said "Failed to load detector" | `core/model_loader.py` exposes `load_detector()` and `load_classifier()`, called separately in `lifespan`; `/health` reports `"degraded"`; the log line says "classifier" |
| 2 | `/analyze` with `{}` or a body that is not JSON answered 500 (Starlette's plain `Internal Server Error`); an empty `image_bytes` raised an OpenCV assertion error (seen in the test run before the guard was added); undecodable bytes answered 500 `Detection failed: 400: Cannot decode image` (the 400 was caught by `except Exception` and re-wrapped), in `/detect` too | 400 with a short reason for each; an empty buffer is treated as not an image; `except HTTPException: raise` before the generic handler |
| 3 | the picture returned by `/detect` (`result_image_b64`) had red and blue swapped (a pure red picture came back as blue), the function was defined twice (the first copy shadowed) and referred to an undefined `logger`; every `/detect` call also wrote `/tmp/api_detection_result.jpg` | one `encode_image_to_base64` without the RGB conversion, `logger` defined; no file written |
| 4 | the rate limit (5 per 60 s) was hard-coded and capped the whole demo behind `Defish-backend` at about five analyses a minute | `rate_limit_requests`, `rate_limit_window` in `Settings` (environment variables `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`); default raised to 60 per 60 s at the owner's request |
| 5 | 500 answers carried the exception text (`Detection failed: <text>`), which can name paths | `Detection failed`; the text goes to the log |
| 6 | no tests (the two `test_*.py` in the root were old scripts: one imported a class that does not exist, the other needed an image that is not in the repository; they were removed, together with `models/onnx_conf.py`, `models/onnx_shape_test.py` and `CI_CD/cron_auto_push.sh`, a server cron job that ran `git add . && git commit && git push` and explains the many "auto commit" entries in the history) | `tests/` with 69 tests, `pytest.ini` (`testpaths = tests`), `requirements-dev.txt` |
| 7 | the crop resize was OpenCV's while training and the reference runtime use Pillow's: confidence of the label differed from the reference by up to 0.12 on 86 crops, 5 labels and 4 `uncertain` flags changed | `_preprocess` resizes with Pillow's bicubic filter; test; now the same labels and flags as the reference, confidences within 1.1e-7 (decision 6) |
| 8 | all crops in one embedder call, 1.3 to 1.9 times slower than one call per crop with the real model | one call per crop (the crop and its mirror image); test; `/analyze` 4.3 / 11.5 / 18.8 s became 3.2 / 5.4 / 12.5 s for 11 / 29 / 46 fish (decision 5) |
| 9 | no `.dockerignore`: `COPY . .` copied `.env`, `.venv` and `models/` into the image (1.64 GB with the models in the build directory) | `.dockerignore` (`.env`, `.venv`, `models`, `tests`, `docs`, `.git`, logs, caches); image 878 MB, no `.env` and an empty `models/` in it (checked by running the image without the compose mounts); compose mounts the checkout, so nothing else changes |

Behaviour a deployer should know: the classifier now resizes crops as in training, so confidences and a few labels and `uncertain` flags differ from the previous service (by up to 0.12 and 5 of 86 crops on the example photos) and `/analyze` is faster; `/health` can now say `"degraded"`; clients that string-match the 500 text of `/detect` or `/analyze` see `Detection failed`; the images returned by `/detect` now have the colours of the photo. Nothing else in a response changed: `/detect` returned exactly the same detections before and after on all four example photos (29, 11, 0 and 46 boxes, every coordinate and score equal).

## Known issues, not changed

| Issue | Where |
|---|---|
| `python main.py` stops with `TypeError: run() got an unexpected keyword argument 'worker_class'`; start with `uvicorn main:app` | `main.py` (`__main__` block) |
| `/detect` runs the detector inside the event loop, blocking every other request while it runs ([measurement](examples/transcripts/health-under-load.txt)) | `api/endpoints.py` |
| after a timeout the work already running in a thread is not stopped | `api/endpoints.py` |
| the Dockerfile starts uvicorn with `--reload`, and the compose file mounts the checkout over `/app`; a development setup in a production unit | `Dockerfile`, `docker-compose.yml` |
| most `Settings` fields are never read; `vds1_ip` is required but unused | `core/config.py` |
| `requirements.txt` lists packages no application code imports (`aiofiles`, `ujson`, `pillow`; `httpx` is used by the tests and the example scripts; `asyncio==4.0.0` installs only package metadata, not a module); `matplotlib` is imported at the top of `inference/detector.py` for a demo function in the same file | `requirements.txt`, `inference/detector.py` |
| the drawing helpers `draw_bboxes_with_labels` and friends in `utils/img.py` are not used by any route; `get_font_scale` prints on every call | `utils/img.py` |
| no CI workflow exists, so the tests have only been run locally | repository |
