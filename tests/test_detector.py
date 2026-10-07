import pytest

from conftest import BOXES, make_models_dir
from core.model_loader import load_detector
from utils.img import letterbox_resize, unletterbox_bbox


@pytest.fixture(scope="module")
def detector(models_dir):
    return load_detector(models_dir)


def run(detector, image):
    """What the API does: letterbox to 960x960, detect, map the boxes back to the original picture."""
    lb, meta = letterbox_resize(image, target_size=960)
    return [dict(d, bbox=unletterbox_bbox(d["bbox"], meta)) for d in detector.predict_image(lb)]


def by_score(found):
    return {round(d["confidence"], 2): d for d in found}


def test_confidence_threshold_and_nms(detector, photo):
    # B duplicates A and is suppressed by NMS (IoU > 0.45); D is below the 0.25 confidence threshold
    assert sorted(by_score(run(detector, photo))) == [0.5, 0.6, 0.9]


def test_boxes_are_mapped_back_to_the_original_resolution(detector, photo):
    found = by_score(run(detector, photo))
    # the 1920x1080 picture is scaled by 0.5 and shifted down by 210 px in the 960x960 frame
    assert found[0.9]["bbox"] == pytest.approx([760, 440, 1160, 640], abs=1e-3)   # A
    assert found[0.5]["bbox"] == pytest.approx([240, 340, 360, 420], abs=1e-3)    # C


def test_box_in_the_padding_collapses_to_a_line(detector, photo):
    x1, y1, x2, y2 = by_score(run(detector, photo))[0.6]["bbox"]                  # E
    assert (x1, x2) == pytest.approx((1340, 1460), abs=1e-3)
    assert y1 == y2 == 0.0


def test_result_format(detector, photo):
    for d in run(detector, photo):
        assert set(d) == {"bbox", "confidence", "class_id"}
        assert d["class_id"] == 0
        assert all(isinstance(v, float) for v in d["bbox"])


def test_no_detection_above_threshold(tmp_path, photo):
    directory = make_models_dir(tmp_path, boxes=[BOXES["D"]], classifier=False)
    assert run(load_detector(directory), photo) == []


def test_load_detector_fails_without_the_file(tmp_path):
    with pytest.raises(Exception):
        load_detector(tmp_path)
