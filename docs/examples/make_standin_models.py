"""Builds tiny stand-in model files, so that the service and its tests run without the real weights.

Neither stand-in knows anything about fish.

* Classifier: the "embedder" is the mean colour of the crop projected to FEATURES numbers by a fixed random matrix,
  and the head is a fixed random linear layer. The predicted class therefore depends only on the average colour of
  the crop, and the confidences carry no information about disease. The file names and array names are the ones
  `inference/classifier.py` reads, so the real loading code runs unchanged.
* Detector: an ONNX graph whose output is a constant YOLOv8-style tensor (1, 5, A) with the boxes given by the caller,
  in 960x960 letterbox coordinates. The tests use it; with `--detector FILE` the command line writes one with three
  fixed boxes that have nothing to do with the picture (the recorded examples of this repository come from the real
  detector, which is not in the repository).

usage:
    python docs/examples/make_standin_models.py models/disease_classifier   # the two files, next to the existing meta.json
    python docs/examples/make_standin_models.py models/disease_classifier --detector models/best.onnx
    python docs/examples/make_standin_models.py --meta OUT_DIR              # also writes a stand-in meta.json
Existing files of the same names are overwritten. Needs `onnx` and `numpy` (requirements-dev.txt).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CLASSES = ["dermatomycosis", "fin_rot", "healthy", "hexamitosis", "mycobacteriosis", "oodiniosis", "plistophorosis"]
FEATURES = 16
EMBEDDER_FILE = "embedder_dinov2_base.onnx"  # the name `meta.json` gives to the real embedder; reused so that meta.json needs no edit
HEAD_FILE = "head_numpy.npz"
DEFAULT_THRESHOLD = 0.83


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def build_embedder(path, seed=0):
    """(N, 3, 224, 224) float -> (N, FEATURES): mean over the image, then a fixed random projection."""
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0, size=(3, FEATURES)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["mean_colour"], axes=[2, 3], keepdims=0),
            helper.make_node("MatMul", ["mean_colour", "projection"], ["embedding"]),
        ],
        "standin_embedder",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", FEATURES])],
        initializer=[numpy_helper.from_array(projection, "projection")],
    )
    _save(graph, path)


def build_head(path, seed=0, scale=3.0):
    """Arrays under the names inference/classifier.py reads. `scale` sets how confident the random head is."""
    rng = np.random.default_rng(seed + 1)
    np.savez(
        path,
        scaler_mean=np.zeros(FEATURES, dtype=np.float32),
        scaler_scale=np.ones(FEATURES, dtype=np.float32),
        lr_coef=rng.normal(0.0, scale, size=(len(CLASSES), FEATURES)).astype(np.float32),
        lr_intercept=np.zeros(len(CLASSES), dtype=np.float32),
    )


def build_meta(path, threshold=DEFAULT_THRESHOLD):
    path.write_text(json.dumps({"embedder": "standin", "classes": CLASSES, "confidence_threshold": threshold,
                                "onnx_embedder": EMBEDDER_FILE}, indent=2))


def build_classifier(out_dir, with_meta=True, threshold=DEFAULT_THRESHOLD, seed=0, scale=3.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_embedder(out_dir / EMBEDDER_FILE, seed)
    build_head(out_dir / HEAD_FILE, seed, scale)
    if with_meta:
        build_meta(out_dir / "meta.json", threshold)


CLI_DETECTOR_BOXES = [(300, 300, 160, 90, 0.88), (620, 420, 120, 70, 0.74), (420, 700, 90, 60, 0.61)]


def build_detector(path, boxes):
    """boxes: list of (cx, cy, w, h, score) in the 960x960 letterbox frame; the graph returns them as (1, 5, len(boxes))."""
    out = np.array(boxes, dtype=np.float32).T[None, ...] if boxes else np.zeros((1, 5, 1), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(out, "boxes"))],
        "standin_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, "height", "width"])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, out.shape[2]])],
    )
    _save(graph, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", help="directory for embedder_dinov2_base.onnx and head_numpy.npz")
    parser.add_argument("--meta", action="store_true", help="also write a stand-in meta.json (default: leave meta.json alone)")
    parser.add_argument("--detector", metavar="FILE", help="also write a stand-in detector (three fixed boxes) to FILE, e.g. models/best.onnx")
    args = parser.parse_args()
    build_classifier(args.out_dir, with_meta=args.meta)
    print(f"stand-in classifier written to {args.out_dir} ({EMBEDDER_FILE}, {HEAD_FILE}); NOT a trained model")
    if args.detector:
        build_detector(args.detector, CLI_DETECTOR_BOXES)
        print(f"stand-in detector written to {args.detector}: three fixed boxes, unrelated to the picture")
