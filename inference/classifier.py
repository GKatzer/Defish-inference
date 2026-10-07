import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class DinoV2Classifier:
    """Disease classifier: frozen DINOv2 embedding + logistic regression, running on onnxruntime only
    (no torch, no scikit-learn at inference time).

    Same predict(image) -> dict interface as the previous YOLO_ONNX_Clf_Inference ("label", "confidence"),
    extended with:
      - "uncertain": True when the top prediction's confidence is below the model's calibrated threshold.
        Replaces the old many_fish/not_a_fish escape classes (dropped -- they were confounding the other
        seven classes and, when a detector crop was bad, the caller had no fallback besides silently
        dropping it). Callers should decide what to show for uncertain==True (e.g. "unclear, consult a
        specialist") rather than trusting the label at face value.
      - "top3": the three most likely labels with their confidences.
    """

    PAD_RGB = (124, 116, 104)  # neutral padding colour, matches training preprocessing
    SIZE = 224

    def __init__(self, artifacts_dir: str, tta: bool = True, chunk_size: int = 1):
        """tta: average the embedding of each crop with that of its mirror image (what the head was evaluated with);
        False halves the embedder work and may move confidences and the uncertain flag.
        chunk_size: crops per embedder call, 0 for all of them in one call."""
        if chunk_size < 0:
            raise ValueError("chunk_size must be 0 (all crops in one call) or a positive number of crops")
        self.tta = tta
        self.chunk_size = chunk_size
        artifacts_dir = Path(artifacts_dir)
        meta = json.loads((artifacts_dir / "meta.json").read_text())
        self.class_names = meta["classes"]
        self.threshold = meta["confidence_threshold"]
        print(f"Class names: {self.class_names}")
        print(f"Confidence threshold (below this -> uncertain): {self.threshold}")
        print(f"Embedder passes per crop: {2 if tta else 1} (mirror image {'on' if tta else 'off'}), "
              f"crops per call: {chunk_size or 'all'}")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(artifacts_dir / meta["onnx_embedder"]), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        head = np.load(artifacts_dir / "head_numpy.npz")
        self.scaler_mean = head["scaler_mean"]
        self.scaler_scale = head["scaler_scale"]
        self.lr_coef = head["lr_coef"]
        self.lr_intercept = head["lr_intercept"]
        print("Classifier loaded (onnxruntime embedder + numpy logistic-regression head)!")

    def _pad_square(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        s = max(h, w)
        canvas = np.full((s, s, 3), self.PAD_RGB, dtype=np.uint8)
        y0, x0 = (s - h) // 2, (s - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = image_rgb
        return canvas

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        square = Image.fromarray(self._pad_square(rgb))
        # Pillow's bicubic filter, as in the training script and the reference runtime (cv2.INTER_CUBIC gives
        # different pixels: confidences differed by up to 0.12 and some labels and uncertain flags changed)
        resized = np.asarray(square.resize((self.SIZE, self.SIZE), Image.BICUBIC))
        return resized.transpose(2, 0, 1).astype(np.float32) / 255.0

    def _embed_batch(self, images_bgr: list) -> np.ndarray:
        """Embeds N crops, `chunk_size` crops per onnxruntime call (default 1: the crop and its horizontal-flip TTA
        pair as a batch of 2; with tta=False a batch of 1; chunk_size=0 puts all N crops in one call).

        One call for all N crops (a (2N, 3, 224, 224) tensor) used to be the way, on the assumption that the
        per-call overhead dominated. Measured with the real embedder (16-thread CPU, onnxruntime 1.23.2) it was
        1.4 to 1.9 times SLOWER than a call per crop: the cost per image grows with the batch, 136 ms at 1 image
        and 191 ms at 32. That was measured on one machine only, which is why the chunk size is a setting.
        Answers are identical for every chunk size.
        """
        n_views = 2 if self.tta else 1
        chunk = self.chunk_size or len(images_bgr)
        feats = []
        for start in range(0, len(images_bgr), chunk):
            views = []
            for img in images_bgr[start:start + chunk]:
                x = self._preprocess(img)                              # (3, H, W)
                views.append(x)
                if self.tta:
                    views.append(x[:, :, ::-1])                        # the mirror image
            out = self.session.run(None, {self.input_name: np.stack(views, axis=0)})[0]   # (chunk * n_views, D)
            feats.append(out.reshape(-1, n_views, out.shape[-1]).mean(axis=1))            # average the views of each crop
        return np.concatenate(feats, axis=0)                           # (N, D)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def predict_batch(self, images: list) -> list:
        """images: list of BGR np.ndarray crops. Returns one result dict per image
        (the embedder runs once per crop, see _embed_batch; the head runs on all of them at once)."""
        if not images:
            return []

        emb = self._embed_batch(images)                                    # (N, D)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)     # L2-normalize
        emb = (emb - self.scaler_mean) / self.scaler_scale                   # StandardScaler
        logits = emb @ self.lr_coef.T + self.lr_intercept                    # (N, C)
        proba = self._softmax(logits)

        results = []
        for row in proba:
            order = np.argsort(-row)
            top1 = int(order[0])
            results.append({
                "label": self.class_names[top1],
                "confidence": float(row[top1]),
                "uncertain": bool(row[top1] < self.threshold),
                "top3": [{"label": self.class_names[i], "confidence": float(row[i])} for i in order[:3]],
            })
        return results

    def predict(self, image: np.ndarray) -> dict:
        """image: BGR np.ndarray, e.g. from utils.img.crop_by_bbox."""
        return self.predict_batch([image])[0]
