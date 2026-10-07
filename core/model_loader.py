from inference.detector import YOLO_ONNX_Inference
from inference.classifier import DinoV2Classifier

MODELS_DIR = "./models"


# The two models are loaded by separate calls, so that a missing file of one of them
# does not take the other one down with it (they used to be built at import time in this module).
def load_detector(models_dir=None):
    models_dir = models_dir or MODELS_DIR
    return YOLO_ONNX_Inference(
        onnx_path=f"{models_dir}/best.onnx",
        img_size=960,
        conf_thres=0.25,
        iou_thres=0.45
    )


def load_classifier(models_dir=None, tta=True, chunk_size=1):
    models_dir = models_dir or MODELS_DIR
    return DinoV2Classifier(
        artifacts_dir=f"{models_dir}/disease_classifier",
        tta=tta,
        chunk_size=chunk_size,
    )