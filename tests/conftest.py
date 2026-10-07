"""Shared fixtures. The tests need no real weights: they run the real loading and inference code on the tiny
stand-in models of docs/examples/make_standin_models.py (the real detector file is not used, so the suite also
passes when models/best.onnx is absent)."""
import base64
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("vds1_ip", "127.0.0.1")  # required by core.config.Settings, not used for inference

_spec = importlib.util.spec_from_file_location("make_standin_models", ROOT / "docs" / "examples" / "make_standin_models.py")
standin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(standin)

# Stand-in detector output in the 960x960 letterbox frame: (cx, cy, w, h, score).
# B duplicates A (suppressed by NMS), D is below the 0.25 confidence threshold, E lies in the padding above a
# 1920x1080 picture (its box collapses to a line when mapped back, so it cannot be cropped).
BOXES = {
    "A": (480, 480, 200, 100, 0.90),
    "B": (484, 482, 196, 100, 0.80),
    "C": (150, 400, 60, 40, 0.50),
    "D": (800, 800, 50, 50, 0.20),
    "E": (700, 100, 60, 40, 0.60),
}


def make_models_dir(path, boxes=BOXES.values(), classifier=True, detector=True, threshold=0.83):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if detector:
        standin.build_detector(path / "best.onnx", list(boxes))
    if classifier:
        standin.build_classifier(path / "disease_classifier", threshold=threshold)
    return path


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    return make_models_dir(tmp_path_factory.mktemp("models"))


@pytest.fixture
def photo():
    """A 1920x1080 picture (BGR): left half dark red, right half dark blue."""
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:, :960] = (30, 30, 200)
    img[:, 960:] = (200, 60, 30)
    return img


def jpeg_bytes(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def b64_jpeg(img):
    return base64.b64encode(jpeg_bytes(img)).decode()


@pytest.fixture
def make_client(monkeypatch):
    """Starts the application (lifespan included) with models loaded from the given directory."""
    from fastapi.testclient import TestClient
    import core.model_loader
    from api.rate_limiter import limiter
    from main import app

    opened = []

    def _make(directory):
        monkeypatch.setattr(core.model_loader, "MODELS_DIR", str(directory))
        limiter.requests.clear()
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        return client

    yield _make
    for client in opened:
        client.__exit__(None, None, None)
    limiter.requests.clear()


@pytest.fixture
def client(make_client, models_dir):
    return make_client(models_dir)
