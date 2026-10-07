import base64
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_np):
    """Converts a numpy array (BGR, as after cv2.imdecode) to a base64 JPEG string"""
    try:
        # cv2.imencode expects BGR, so no conversion to RGB
        _, buffer = cv2.imencode('.jpg', image_np)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

def letterbox_resize(image: np.ndarray, target_size: int = 960, color=(255, 255, 255)):
    """
    Fits the image into a square target_size x target_size without distorting the proportions.
    The free margins are filled with the colour color (white by default).

    Returns (canvas, meta). meta is needed by unletterbox_bbox to convert the
    bbox from the canvas coordinates back to the coordinates of the original image.
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    meta = {
        "scale": scale,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "orig_w": w,
        "orig_h": h,
    }
    return canvas, meta

def unletterbox_bbox(bbox, meta):
    """Converts a bbox from the letterbox canvas coordinates back to the coordinates of the original image."""
    x1, y1, x2, y2 = bbox
    scale = meta["scale"]
    x_offset = meta["x_offset"]
    y_offset = meta["y_offset"]

    x1 = (x1 - x_offset) / scale
    y1 = (y1 - y_offset) / scale
    x2 = (x2 - x_offset) / scale
    y2 = (y2 - y_offset) / scale

    x1 = float(np.clip(x1, 0, meta["orig_w"]))
    y1 = float(np.clip(y1, 0, meta["orig_h"]))
    x2 = float(np.clip(x2, 0, meta["orig_w"]))
    y2 = float(np.clip(y2, 0, meta["orig_h"]))

    return [x1, y1, x2, y2]

def crop_by_bbox(image: np.ndarray, bbox, pad: int = 0):
    """
    image: BGR np.ndarray
    bbox: [x1, y1, x2, y2]
    pad: padding in pixels
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]

def put_text_with_outline(img, text, pos, font_scale):
    x, y = pos

    # black outline
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0,0,0), 4, cv2.LINE_AA)

def get_font_scale(x1, y1, x2, y2, img_h):
    box_h = abs(y2 - y1)

    # based on the bbox size + the image
    scale = img_h / 1000

    print("scale:", scale)

    return float(np.clip(scale, 1.0, 4.0))

def draw_bboxes_with_labels(image, results):
    """
    results = [{
        "bbox": [x1, y1, x2, y2],
        "det_confidence": float,
        "class": str,
        "class_confidence": float (optional)
    }]
    """
    vis = image.copy()


    for res in results:
        x1, y1, x2, y2 = map(int, res["bbox"])

        h_img = vis.shape[0]
        font_scale = get_font_scale(x1, y1, x2, y2, h_img)

        label = res["class"]
        det_conf = res["det_confidence"]
        text = f"{label} ({det_conf:.2f})"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, 2)

        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)

        put_text_with_outline(vis, text, (x1 + 2, y1 - 4), font_scale)

    return vis
