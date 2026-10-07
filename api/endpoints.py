# api/endpoints.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from .rate_limiter import limiter
import asyncio
import numpy as np
import cv2
import logging
import base64
import io
from core.globals import models
from utils.img import encode_image_to_base64, crop_by_bbox, letterbox_resize, unletterbox_bbox

MAX_SIDE = 960
REQ_TIMEOUT = 300.0  # 180 s was too close to the 170 s a 29-fish photo took on a 2-core Xeon VDS

logger = logging.getLogger(__name__)

router = APIRouter()

def _decode_image(data: bytes):
    """BGR array, or None when the bytes are not an image (OpenCV raises on an empty buffer)."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Inference Service",
        "docs": "/docs",
        "health": "/health",
        "detector": "loaded" if models.get("detector") else "not loaded",
        "classifier": "loaded" if models.get("classifier") else "failed",
    }

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    detector_status = "loaded" if models.get("detector") else "failed"
    classifier_status = "loaded" if models.get("classifier") else "failed"
    both_loaded = detector_status == "loaded" and classifier_status == "loaded"
    return {
        "status": "healthy" if both_loaded else "degraded",
        "service": "ml-inference",
        "version": "1.0.0",
        "detector": detector_status,
        "classifier": classifier_status,
    }

@router.post("/detect")
async def detect_fish(
    request: Request,
    image: UploadFile = File(...)
):
    """Fish detection on an image through ONNX Runtime"""
    
    # Rate limiting by IP
    client_ip = request.client.host if request.client else "unknown"
    limiter.check_limit(client_ip)
    
    detector = models.get("detector")
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector model not loaded")
    
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        # Timeout for the processing
        result = await asyncio.wait_for(
            _detect_fish_internal(image),
            timeout=REQ_TIMEOUT
        )
        return result

    except asyncio.TimeoutError:
        logger.error(f"Timeout processing image from {client_ip}")
        raise HTTPException(
            status_code=408,
            detail="Processing timeout. Image might be too large."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Detection failed")

async def _detect_fish_internal(image: UploadFile):
    """Internal detection function"""
    image_bytes = await image.read()

    np_img_orig = _decode_image(image_bytes)

    if np_img_orig is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    logger.info(f"Image loaded: shape={np_img_orig.shape}")

    np_img_lb, lb_meta = letterbox_resize(np_img_orig, target_size=MAX_SIDE)

    detections = models["detector"].predict_image(np_img_lb)

    # Convert the bbox back to the coordinates of the original (not downscaled) image
    formatted_detections = []
    for det in detections:
        formatted_detections.append({
            "bbox": unletterbox_bbox(det["bbox"], lb_meta),
            "confidence": float(det["confidence"]),
            "class_id": int(det["class_id"])
        })

    logger.info(f"Objects detected: {len(formatted_detections)}")
    for det in formatted_detections:
        logger.info(f"Detection: {det}")

    result_image = models["detector"].draw_detections(np_img_orig.copy(), formatted_detections)

    result_image_b64 = encode_image_to_base64(result_image)

    return {
        "status": "success",
        "count": len(formatted_detections),
        "detections": formatted_detections,
        "result_image_b64": result_image_b64 if result_image_b64 else None,
        "message": f"Found {len(formatted_detections)} objects"
    }

@router.post("/analyze")
async def analyze_image(
    request: Request,):
    """Full analysis of an image: detection and classification of each object"""
    
    # Rate limiting by IP
    client_ip = request.client.host if request.client else "unknown"
    limiter.check_limit(client_ip)
    
    try:
        body = await request.json()
        image_bytes = base64.b64decode(body["image_bytes"])
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Body must be JSON with a base64 string in 'image_bytes'"
        )

    np_img = _decode_image(image_bytes)
    if np_img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    # Timeout for the processing
    try:
        result = await asyncio.wait_for(
            _analyze_image_internal(
                np_img
            ),
            timeout=REQ_TIMEOUT
        )
        return result

    except asyncio.TimeoutError:
        logger.error(f"Timeout processing image from {client_ip}")
        raise HTTPException(
            status_code=408,
            detail="Processing timeout. Image might be too large."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Detection failed")

async def _analyze_image_internal(np_img):
    """Internal analysis function"""
    detector = models.get("detector")
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector model not loaded")

    classifier = models.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier model not loaded")

    if np_img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    
    logger.info(f"Image loaded: shape={np_img.shape}")

    # Detection runs on the letterbox version (960x960, fixed size): fast and without distorting the proportions.
    # The crop for the classifier and the final picture are made at the original resolution.
    np_img_lb, lb_meta = letterbox_resize(np_img, target_size=MAX_SIDE)

    detections = await run_in_threadpool(detector.predict_image, np_img_lb)

    logger.info(f"Objects detected: {len(detections)}")

    # Crop every detection first, then classify the crops. By default the embedder runs once per crop (a batch of
    # the crop and its mirror image): one call for all crops was measured 1.4-1.9x slower on a 16-thread CPU.
    # The mirror pass and the crops per call are settings (CLASSIFIER_TTA, CLASSIFIER_CHUNK_SIZE).
    boxes = []
    crops = []
    for det in detections:
        bbox = unletterbox_bbox(det["bbox"], lb_meta)
        crop = crop_by_bbox(np_img, bbox)
        if crop is None:
            continue
        boxes.append((bbox, det["confidence"]))
        crops.append(crop)

    classifications = await run_in_threadpool(classifier.predict_batch, crops)

    results = []
    uncertain_counter = 0

    for (bbox, det_confidence), clas in zip(boxes, classifications):
        if clas["uncertain"]:
            uncertain_counter += 1

        # Every detector box gets a result now -- the classifier no longer has many_fish/not_a_fish
        # escape classes to silently swallow a bad crop into. Low-confidence calls are still returned,
        # flagged via "uncertain", so the caller can decide how to present them (e.g. "unclear, consult
        # a specialist") instead of the diagnosis silently vanishing.
        results.append({
            "bbox": bbox,
            "det_confidence": det_confidence,
            "class": clas["label"],
            "class_confidence": clas["confidence"],
            "uncertain": clas["uncertain"],
            "top3": clas["top3"],
        })

    _, buffer = cv2.imencode(".jpg", np_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    logger.info(f"Results: {results}")
    logger.info(f"Uncertain predictions: {uncertain_counter}/{len(results)}")

    h, w = np_img.shape[:2]

    return {
        "detections": results,
        "image": img_base64,
        "image_width": w,
        "image_height": h
    }

@router.get("/test")
async def test_endpoint():
    """Test endpoint for checking that the API works"""
    return {"message": "API is working", "status": "ok"}

@router.get("/model-status")
async def model_status():
    """Check the status of the loaded models"""
    status = {}
    for model_name, model in models.items():
        status[model_name] = {
            "loaded": model is not None,
            "type": type(model).__name__ if model else None
        }
    return status