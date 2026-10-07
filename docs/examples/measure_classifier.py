"""Times the classifier on the crops the detector finds, and checks it against two other ways of computing the same thing.
Needs the REAL model files in ./models (detector and classifier); run it from the repository root.

For every photo it: detects fish and crops them from the original (as /analyze does), then
  1. times the current `classifier.predict_batch(crops)` (one embedder call per crop, the crop and its mirror image as
     a batch of 2),
  2. times the PREVIOUS way, kept here for comparison: one embedder call for all crops (a batch of 2N), and checks that
     the answers are the same,
  3. with --reference FILE (the `infer_onnx.py` of Defish-ML-train, which resizes with Pillow's bicubic filter and runs
     the original and the mirror image as two separate calls), compares labels and confidences with it.

Options: --threads N sets ONNX Runtime's intra-op thread count for the embedder (default: ONNX Runtime's own choice);
--opencv-resize puts back the crop resize the service used before (OpenCV's cubic interpolation instead of Pillow's
bicubic filter), to show how much that alone moved the answers against the reference; --scan also times the embedder
alone for batches of 1 to 32 images (real preprocessed crops of the first photo with at least 32 fish).

usage: python docs/examples/measure_classifier.py docs/examples/photos/*.jpg [--reference PATH/TO/infer_onnx.py] [--runs 3]
                                                  [--threads N] [--opencv-resize] [--scan]
"""
import argparse
import importlib.util
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from core.model_loader import load_classifier, load_detector  # noqa: E402
from utils.img import crop_by_bbox, letterbox_resize, unletterbox_bbox  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("photos", nargs="+")
parser.add_argument("--reference")
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--threads", type=int)
parser.add_argument("--opencv-resize", action="store_true")
parser.add_argument("--scan", action="store_true")
args = parser.parse_args()

detector, classifier = load_detector(), load_classifier()
if args.threads:
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    classifier.session = ort.InferenceSession("models/disease_classifier/embedder_dinov2_base.onnx", options, providers=["CPUExecutionProvider"])
if args.opencv_resize:
    def _preprocess_opencv(image_bgr):
        square = classifier._pad_square(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        resized = cv2.resize(square, (classifier.SIZE, classifier.SIZE), interpolation=cv2.INTER_CUBIC)
        return resized.transpose(2, 0, 1).astype(np.float32) / 255.0

    classifier._preprocess = _preprocess_opencv

reference = None
if args.reference:
    spec = importlib.util.spec_from_file_location("reference_infer_onnx", args.reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.FishDiseaseClassifierONNX("models/disease_classifier")

current_embed = classifier._embed_batch


def one_call_for_all(images_bgr):
    """The previous implementation: all crops and their mirror images in one (2N, 3, 224, 224) call."""
    originals = np.stack([classifier._preprocess(img) for img in images_bgr], axis=0)
    batch = np.concatenate([originals, originals[:, :, :, ::-1]], axis=0)
    feats = classifier.session.run(None, {classifier.input_name: batch})[0]
    n = len(images_bgr)
    return (feats[:n] + feats[n:]) / 2.0


def timed(fn):
    started = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - started) * 1000


def predict_with(embed, crops):
    classifier._embed_batch = embed
    try:
        return classifier.predict_batch(crops)
    finally:
        classifier._embed_batch = current_embed


header = (f"{'photo':34} {'crops':>5} {'per crop ms':>12} {'one call ms':>12} {'one call / per crop':>20} "
          f"{'same label':>11} {'max |dconf|':>12}")
if reference:
    header += f" {'ref same label':>15} {'ref max |dconf|':>16} {'ref flips gate':>15}"
print(header)
for path in args.photos:
    image = cv2.imread(path)
    canvas, meta = letterbox_resize(image, target_size=960)
    boxes = [unletterbox_bbox(d["bbox"], meta) for d in detector.predict_image(canvas)]
    crops = [c for c in (crop_by_bbox(image, b) for b in boxes) if c is not None]
    if not crops:
        print(f"{os.path.basename(path):34} {0:5d}   (no fish found, nothing to classify)")
        continue
    classifier.predict_batch(crops)  # warm-up
    per_crop_times, one_call_times = [], []
    for _ in range(args.runs):
        current, ms = timed(lambda: classifier.predict_batch(crops))
        per_crop_times.append(ms)
        previous, ms = timed(lambda: predict_with(one_call_for_all, crops))
        one_call_times.append(ms)
    per_crop_ms, one_call_ms = statistics.median(per_crop_times), statistics.median(one_call_times)
    same = sum(a["label"] == b["label"] for a, b in zip(current, previous))
    dconf = max(abs(a["confidence"] - b["confidence"]) for a, b in zip(current, previous))
    row = (f"{os.path.basename(path):34} {len(crops):5d} {per_crop_ms:12.0f} {one_call_ms:12.0f} {one_call_ms / per_crop_ms:20.2f} "
           f"{f'{same}/{len(crops)}':>11} {dconf:12.2e}")
    if reference:
        refs = [reference.predict(c) for c in crops]
        ref_same = sum(a["label"] == r["label"] for a, r in zip(current, refs))
        ref_dconf = max(abs(a["confidence"] - r["confidence"]) for a, r in zip(current, refs))
        flips = sum(a["uncertain"] != r["uncertain"] for a, r in zip(current, refs))
        row += f" {f'{ref_same}/{len(crops)}':>15} {ref_dconf:16.2e} {flips:15d}"
    print(row)

if args.scan:
    for path in args.photos:
        image = cv2.imread(path)
        canvas, meta = letterbox_resize(image, target_size=960)
        crops = [c for c in (crop_by_bbox(image, unletterbox_bbox(d["bbox"], meta)) for d in detector.predict_image(canvas)) if c is not None]
        if len(crops) < 32:
            continue
        batch = np.stack([classifier._preprocess(c) for c in crops[:32]])
        print(f"\nembedder alone on real crops of {os.path.basename(path)} (median of 3 runs after a warm-up):")
        for size in (1, 2, 4, 8, 16, 32):
            classifier.session.run(None, {classifier.input_name: batch[:size]})
            times = []
            for _ in range(3):
                started = time.perf_counter()
                classifier.session.run(None, {classifier.input_name: batch[:size]})
                times.append((time.perf_counter() - started) * 1000)
            ms = statistics.median(times)
            print(f"  batch of {size:2d} images: {ms:7.0f} ms = {ms / size:5.0f} ms per image")
        break
