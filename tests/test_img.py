import base64

import cv2
import numpy as np
import pytest

from utils.img import crop_by_bbox, encode_image_to_base64, letterbox_resize, unletterbox_bbox


def test_letterbox_is_square_and_keeps_proportions():
    img = np.zeros((1080, 1920, 3), np.uint8)
    canvas, meta = letterbox_resize(img, target_size=960)
    assert canvas.shape == (960, 960, 3)
    assert meta["scale"] == 0.5
    assert (meta["x_offset"], meta["y_offset"]) == (0, 210)
    assert (meta["orig_w"], meta["orig_h"]) == (1920, 1080)


def test_letterbox_pads_with_white_and_keeps_picture_in_the_middle():
    img = np.zeros((100, 200, 3), np.uint8)  # black
    canvas, meta = letterbox_resize(img, target_size=100)
    assert (canvas[0, 0] == 255).all()                      # padding, top-left corner
    assert (canvas[50, 50] == 0).all()                      # the picture
    assert meta["y_offset"] == 25


def test_letterbox_portrait_offsets_x():
    img = np.zeros((200, 100, 3), np.uint8)
    _, meta = letterbox_resize(img, target_size=100)
    assert (meta["x_offset"], meta["y_offset"]) == (25, 0)


@pytest.mark.parametrize("shape", [(1080, 1920), (200, 100), (960, 960), (480, 640)])
def test_unletterbox_inverts_letterbox(shape):
    h, w = shape
    _, meta = letterbox_resize(np.zeros((h, w, 3), np.uint8), target_size=960)
    # a box given in the original picture, mapped forward by hand, comes back to itself
    original = [w * 0.25, h * 0.25, w * 0.75, h * 0.5]
    forward = [original[0] * meta["scale"] + meta["x_offset"], original[1] * meta["scale"] + meta["y_offset"],
               original[2] * meta["scale"] + meta["x_offset"], original[3] * meta["scale"] + meta["y_offset"]]
    back = unletterbox_bbox(forward, meta)
    assert back == pytest.approx(original, abs=1e-6)


def test_unletterbox_clips_to_the_original_picture():
    _, meta = letterbox_resize(np.zeros((1080, 1920, 3), np.uint8), target_size=960)
    assert unletterbox_bbox([-50, 0, 2000, 960], meta) == [0.0, 0.0, 1920.0, 1080.0]


def test_crop_by_bbox_cuts_and_clips():
    img = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    assert crop_by_bbox(img, [10, 20, 50, 60]).shape == (40, 40, 3)
    assert crop_by_bbox(img, [-5, -5, 20, 20]).shape == (20, 20, 3)         # clipped to the picture
    assert crop_by_bbox(img, [150, 50, 400, 400]).shape == (50, 50, 3)
    assert crop_by_bbox(img, [10, 10, 10, 50]) is None                      # no width
    assert crop_by_bbox(img, [300, 10, 400, 50]) is None                    # entirely outside


def test_encode_image_to_base64_keeps_colours():
    """A BGR picture must come back with the same channels (it used to be returned with red and blue swapped)."""
    red_bgr = np.zeros((20, 20, 3), np.uint8)
    red_bgr[:] = (0, 0, 255)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encode_image_to_base64(red_bgr)), np.uint8), cv2.IMREAD_COLOR)
    b, g, r = (int(v) for v in decoded[10, 10])
    assert r > 240 and b < 15 and g < 15


def test_encode_image_to_base64_returns_none_on_bad_input():
    assert encode_image_to_base64(None) is None
