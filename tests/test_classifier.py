import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import standin
from core.model_loader import load_classifier


@pytest.fixture(scope="module")
def classifier(models_dir):
    return load_classifier(models_dir)


def crop(colour_bgr, h=60, w=40):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = colour_bgr
    return img


CROPS = [crop((30, 30, 200)), crop((200, 60, 30), 30, 90), crop((20, 200, 20), 50, 50), crop((90, 90, 90), 70, 20)]


def test_batch_equals_one_by_one(classifier):
    batch = classifier.predict_batch(CROPS)
    for single, from_batch in zip((classifier.predict(c) for c in CROPS), batch):
        assert single["label"] == from_batch["label"]
        assert single["confidence"] == pytest.approx(from_batch["confidence"], abs=1e-5)


def test_empty_batch(classifier):
    assert classifier.predict_batch([]) == []


def test_result_fields_and_top3(classifier):
    for r in classifier.predict_batch(CROPS):
        assert set(r) == {"label", "confidence", "uncertain", "top3"}
        assert r["label"] in standin.CLASSES
        assert len(r["top3"]) == 3
        confidences = [t["confidence"] for t in r["top3"]]
        assert confidences == sorted(confidences, reverse=True)
        assert r["top3"][0] == {"label": r["label"], "confidence": r["confidence"]}
        assert len({t["label"] for t in r["top3"]}) == 3


def test_confidences_are_probabilities(classifier):
    # the three shown are part of a softmax over seven classes: positive and their sum cannot exceed 1
    for r in classifier.predict_batch(CROPS):
        total = sum(t["confidence"] for t in r["top3"])
        assert 0 < r["confidence"] <= 1 and total <= 1 + 1e-6


@pytest.mark.parametrize("threshold, expected", [(0.0, False), (1.01, True)])
def test_gate_follows_the_threshold(tmp_path, threshold, expected):
    standin.build_classifier(tmp_path / "disease_classifier", threshold=threshold)
    results = load_classifier(tmp_path).predict_batch(CROPS)
    assert {r["uncertain"] for r in results} == {expected}


def test_gate_is_strictly_below_the_threshold(tmp_path):
    """uncertain is `confidence < threshold`: a prediction exactly at the threshold is kept."""
    standin.build_classifier(tmp_path / "a" / "disease_classifier", threshold=0.5)
    first = load_classifier(tmp_path / "a").predict(CROPS[0])
    standin.build_classifier(tmp_path / "b" / "disease_classifier", threshold=first["confidence"])
    assert load_classifier(tmp_path / "b").predict(CROPS[0])["uncertain"] is False


def test_mirrored_crop_gives_the_same_answer(classifier):
    # flip augmentation at inference: the embedding is the mean over the crop and its mirror image
    for c in CROPS:
        a, b = classifier.predict(c), classifier.predict(c[:, ::-1].copy())
        assert a["label"] == b["label"] and a["confidence"] == pytest.approx(b["confidence"], abs=1e-5)


def test_preprocess_pads_to_a_square_with_the_neutral_colour(classifier):
    tensor = classifier._preprocess(crop((0, 0, 0), h=10, w=30))        # BGR black crop
    assert tensor.shape == (3, 224, 224) and tensor.dtype == np.float32
    expected_rgb = np.array(classifier.PAD_RGB, dtype=np.float32) / 255.0
    assert tensor[:, 2, 112] == pytest.approx(expected_rgb, abs=0.03)    # padding above the crop (cubic resize may overshoot)
    assert tensor[:, 112, 112] == pytest.approx([0, 0, 0], abs=0.03)      # the crop itself, in the middle


def test_load_classifier_fails_without_the_files(tmp_path):
    with pytest.raises(Exception):
        load_classifier(tmp_path)


def test_crop_resize_is_pillows_bicubic_as_in_training(classifier):
    """Training and the reference runtime resize with Pillow's BICUBIC; OpenCV's INTER_CUBIC gives different pixels
    (with the real model that moved confidences by up to 0.12 and changed some labels and gate flags)."""
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(50, 80, 3), dtype=np.uint8)      # BGR crop
    rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    square = classifier._pad_square(rgb)
    expected = np.asarray(Image.fromarray(square).resize((224, 224), Image.BICUBIC)).transpose(2, 0, 1).astype(np.float32) / 255.0
    assert np.array_equal(classifier._preprocess(noisy), expected)
    opencv = cv2.resize(square, (224, 224), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1).astype(np.float32) / 255.0
    assert np.abs(opencv - expected).max() > 1 / 255                    # the two really differ, so this test can fail


def test_embedder_runs_once_per_crop_on_the_crop_and_its_mirror_image(classifier, monkeypatch):
    """One call for all crops was measured 1.4 to 1.9 times slower than one call per crop with the real embedder."""
    calls = []

    class Spy:
        def __init__(self, session):
            self.session = session

        def run(self, outputs, feed):
            (batch,) = feed.values()
            calls.append(batch)
            return self.session.run(outputs, feed)

    monkeypatch.setattr(classifier, "session", Spy(classifier.session))
    classifier.predict_batch(CROPS)
    assert len(calls) == len(CROPS)
    for batch in calls:
        assert batch.shape == (2, 3, 224, 224)
        assert np.array_equal(batch[1], batch[0][:, :, ::-1])           # the second item is the mirror image


class Spy:
    """Wraps the embedder session and records the batch of every call."""

    def __init__(self, session):
        self.session = session
        self.calls = []

    def run(self, outputs, feed):
        (batch,) = feed.values()
        self.calls.append(batch)
        return self.session.run(outputs, feed)


def spied(models_dir, **options):
    clf = load_classifier(models_dir, **options)
    clf.session = Spy(clf.session)
    return clf


def test_without_tta_the_embedder_sees_one_image_per_crop(models_dir):
    clf = spied(models_dir, tta=False)
    clf.predict_batch(CROPS)
    assert [c.shape for c in clf.session.calls] == [(1, 3, 224, 224)] * len(CROPS)


@pytest.mark.parametrize("chunk_size, expected_batches", [(1, [2, 2, 2, 2]), (3, [6, 2]), (4, [8]), (10, [8]), (0, [8])])
def test_chunk_size_sets_the_crops_per_embedder_call(models_dir, chunk_size, expected_batches):
    """chunk_size=0 means all crops in one call; a chunk larger than the number of crops is the same thing."""
    clf = spied(models_dir, chunk_size=chunk_size)
    clf.predict_batch(CROPS)
    assert [len(c) for c in clf.session.calls] == expected_batches


@pytest.mark.parametrize("options", [{"chunk_size": 3}, {"chunk_size": 0}, {"tta": False}, {"tta": False, "chunk_size": 0}])
def test_options_do_not_change_the_answers_of_the_stand_in(models_dir, classifier, options):
    """The stand-in embedder only sees the mean colour, so the mirror image and the grouping cannot matter; this
    checks that the chunks are cut and reassembled in the right order (the four crops have different colours)."""
    expected = classifier.predict_batch(CROPS)
    for want, got in zip(expected, load_classifier(models_dir, **options).predict_batch(CROPS)):
        assert want["label"] == got["label"]
        assert want["confidence"] == pytest.approx(got["confidence"], abs=1e-5)


def test_mirror_image_pairs_stay_with_their_crop_in_a_chunk(models_dir):
    clf = spied(models_dir, chunk_size=0)
    clf.predict_batch(CROPS)
    (batch,) = clf.session.calls
    for i in range(len(CROPS)):
        assert np.array_equal(batch[2 * i + 1], batch[2 * i][:, :, ::-1])


def test_negative_chunk_size_is_rejected(models_dir):
    with pytest.raises(ValueError):
        load_classifier(models_dir, chunk_size=-1)


def test_the_settings_default_to_the_recorded_behaviour():
    from core.config import settings
    assert settings.classifier_tta is True and settings.classifier_chunk_size == 1
