import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import api.endpoints
from conftest import BOXES, b64_jpeg, jpeg_bytes, make_models_dir, standin
from core.globals import models


# ---- health and status -------------------------------------------------------------------------------

def test_health_when_both_models_are_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "service": "ml-inference", "version": "1.0.0", "detector": "loaded", "classifier": "loaded"}


def test_model_status_names_the_loaded_classes(client):
    assert client.get("/model-status").json() == {
        "detector": {"loaded": True, "type": "YOLO_ONNX_Inference"},
        "classifier": {"loaded": True, "type": "DinoV2Classifier"},
    }


def test_root_and_test_endpoints(client):
    assert client.get("/").json()["detector"] == "loaded"
    assert client.get("/test").json() == {"message": "API is working", "status": "ok"}


def test_missing_classifier_files_leave_the_detector_loaded(make_client, tmp_path):
    """The detector and the classifier are loaded separately; a missing classifier used to take the detector down too."""
    client = make_client(make_models_dir(tmp_path, classifier=False))
    assert client.get("/health").json() == {"status": "degraded", "service": "ml-inference", "version": "1.0.0",
                                            "detector": "loaded", "classifier": "failed"}
    assert client.get("/model-status").json()["detector"]["loaded"] is True


def test_missing_detector_file_leaves_the_classifier_loaded(make_client, tmp_path):
    client = make_client(make_models_dir(tmp_path, detector=False))
    body = client.get("/health").json()
    assert (body["status"], body["detector"], body["classifier"]) == ("degraded", "failed", "loaded")


def test_nothing_loaded(make_client, tmp_path):
    client = make_client(tmp_path)
    body = client.get("/health").json()
    assert (body["status"], body["detector"], body["classifier"]) == ("degraded", "failed", "failed")


def test_models_are_released_on_shutdown(make_client, models_dir):
    client = make_client(models_dir)
    assert set(models) == {"detector", "classifier"}
    client.__exit__(None, None, None)
    assert models == {}
    client.__enter__()  # fixture teardown closes it again


# ---- /detect --------------------------------------------------------------------------------------------

def test_detect_returns_boxes_in_original_coordinates(client, photo):
    r = client.post("/detect", files={"image": ("p.jpg", jpeg_bytes(photo), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" and body["count"] == 3 and body["message"] == "Found 3 objects"
    boxes = {round(d["confidence"], 2): d["bbox"] for d in body["detections"]}
    assert boxes[0.9] == pytest.approx([760, 440, 1160, 640], abs=1e-3)
    assert all(d["class_id"] == 0 for d in body["detections"])


def test_detect_works_without_the_classifier(make_client, tmp_path, photo):
    client = make_client(make_models_dir(tmp_path, classifier=False))
    r = client.post("/detect", files={"image": ("p.jpg", jpeg_bytes(photo), "image/jpeg")})
    assert r.status_code == 200 and r.json()["count"] == 3


def test_detect_result_image_keeps_the_colours(client):
    red = np.zeros((100, 200, 3), np.uint8)
    red[:] = (0, 0, 255)  # BGR red
    r = client.post("/detect", files={"image": ("red.jpg", jpeg_bytes(red), "image/jpeg")})
    img = cv2.imdecode(np.frombuffer(base64.b64decode(r.json()["result_image_b64"]), np.uint8), cv2.IMREAD_COLOR)
    b, g, rr = (int(v) for v in img[90, 20])  # a corner, away from the boxes the detector draws
    assert rr > 200 and b < 60  # red stays red (the answer used to have red and blue swapped)


def test_detect_does_not_write_the_result_to_disk(client, photo, monkeypatch):
    written = []
    monkeypatch.setattr(api.endpoints.cv2, "imwrite", lambda *a, **k: written.append(a) or True)
    client.post("/detect", files={"image": ("p.jpg", jpeg_bytes(photo), "image/jpeg")})
    assert written == []


def test_detect_rejects_a_file_that_is_not_an_image_type(client):
    r = client.post("/detect", files={"image": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and r.json() == {"detail": "Uploaded file must be an image"}


def test_detect_answers_400_for_bytes_that_do_not_decode(client):
    """It used to answer 500 'Detection failed: 400: Cannot decode image'."""
    r = client.post("/detect", files={"image": ("x.jpg", b"this is not an image", "image/jpeg")})
    assert r.status_code == 400 and r.json() == {"detail": "Cannot decode image"}


def test_detect_answers_400_for_an_empty_file(client):
    r = client.post("/detect", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400 and r.json() == {"detail": "Cannot decode image"}


def test_detect_requires_the_image_field(client):
    assert client.post("/detect", files={"other": ("p.jpg", b"x", "image/jpeg")}).status_code == 422


# ---- /analyze -------------------------------------------------------------------------------------------

def analyze(client, photo):
    return client.post("/analyze", json={"image_bytes": b64_jpeg(photo)})


def test_analyze_returns_one_result_per_croppable_box(client, photo):
    r = analyze(client, photo)
    assert r.status_code == 200
    body = r.json()
    assert (body["image_width"], body["image_height"]) == (1920, 1080)
    # three boxes pass the detector; the one in the padding collapses to a line and cannot be cropped
    assert len(body["detections"]) == 2
    for d in body["detections"]:
        assert set(d) == {"bbox", "det_confidence", "class", "class_confidence", "uncertain", "top3"}
        assert d["class"] in standin.CLASSES and isinstance(d["uncertain"], bool)
        assert d["top3"][0]["label"] == d["class"] and len(d["top3"]) == 3
    assert sorted(round(d["det_confidence"], 2) for d in body["detections"]) == [0.5, 0.9]


def test_analyze_returns_the_original_picture_reencoded(client, photo):
    body = analyze(client, photo).json()
    img = cv2.imdecode(np.frombuffer(base64.b64decode(body["image"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == photo.shape
    assert tuple(int(v) for v in img[540, 100]) == pytest.approx((30, 30, 200), abs=12)  # colours kept


def test_analyze_with_no_fish(make_client, tmp_path, photo):
    client = make_client(make_models_dir(tmp_path, boxes=[BOXES["D"]]))
    body = analyze(client, photo).json()
    assert body["detections"] == [] and body["image_width"] == 1920


def test_analyze_gate_flag_reaches_the_answer(make_client, tmp_path, photo):
    always_unsure = make_client(make_models_dir(tmp_path / "unsure", threshold=1.01))
    assert {d["uncertain"] for d in analyze(always_unsure, photo).json()["detections"]} == {True}


@pytest.mark.parametrize("body, kind", [
    ({}, "json"),
    ({"image": "x"}, "json"),
    ({"image_bytes": 123}, "json"),
    ({"image_bytes": "abc"}, "json"),                      # invalid base64 padding
    ({"image_bytes": base64.b64encode(b"not an image").decode()}, "decode"),
    ({"image_bytes": ""}, "decode"),                       # an empty buffer makes cv2.imdecode raise, it used to end in a 500
])
def test_analyze_bad_input_is_400(client, body, kind):
    r = client.post("/analyze", json=body)
    assert r.status_code == 400
    assert r.json()["detail"] == ("Cannot decode image" if kind == "decode" else "Body must be JSON with a base64 string in 'image_bytes'")


def test_analyze_body_that_is_not_json_is_400(client):
    r = client.post("/analyze", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_analyze_without_the_classifier_is_500_with_the_reason(make_client, tmp_path, photo):
    client = make_client(make_models_dir(tmp_path, classifier=False))
    r = analyze(client, photo)
    assert r.status_code == 500 and r.json() == {"detail": "Classifier model not loaded"}


def test_analyze_without_the_detector_is_500_with_the_reason(make_client, tmp_path, photo):
    client = make_client(make_models_dir(tmp_path, detector=False))
    r = analyze(client, photo)
    assert r.status_code == 500 and r.json() == {"detail": "Detector model not loaded"}


def test_internal_errors_do_not_leak_exception_text(client, photo, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret path /srv/models/best.onnx")
    monkeypatch.setattr(models["detector"], "predict_image", boom)
    r = analyze(client, photo)
    assert r.status_code == 500 and r.json() == {"detail": "Detection failed"}


def test_analyze_times_out_with_408(client, photo, monkeypatch):
    monkeypatch.setattr(api.endpoints, "REQ_TIMEOUT", 0.05)
    monkeypatch.setattr(models["detector"], "predict_image", lambda img: time.sleep(0.5) or [])
    r = analyze(client, photo)
    assert r.status_code == 408 and "timeout" in r.json()["detail"].lower()


# ---- rate limit -----------------------------------------------------------------------------------------

def test_the_61st_request_in_a_minute_is_refused_by_default(client):
    statuses = [client.post("/analyze", json={}).status_code for _ in range(62)]  # failing requests count too
    assert statuses == [400] * 60 + [429] * 2
    r = client.post("/analyze", json={})
    detail = r.json()["detail"]
    assert detail["error"] == "Too many requests" and detail["limit"] == 60 and detail["window"] == 60


def test_the_limit_applies_to_detect_and_analyze_together(client, monkeypatch):
    from api.rate_limiter import limiter
    monkeypatch.setattr(limiter, "max_requests", 5)
    for _ in range(3):
        client.post("/analyze", json={})
    for _ in range(2):
        client.post("/detect", files={"image": ("a.txt", b"x", "text/plain")})
    assert client.post("/detect", files={"image": ("a.txt", b"x", "text/plain")}).status_code == 429


def test_health_is_not_rate_limited(client):
    assert all(client.get("/health").status_code == 200 for _ in range(20))
