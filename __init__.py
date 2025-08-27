"""
Roboflow RF-DETR models.
"""

import eta.core.web as etaw

from fiftyone.operators import types
import fiftyone.utils.ultralytics as fouu


def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download.
        model_path: the path to download the model to.
    """
    url = MODEL_URLS[model_name]
    etaw.download_file(url, path=model_path)


def load_model(model_name, model_path, classes=None):
    """Loads the model.

    Args:
        model_name: the name of the model to load.
        model_path: the path to download the model to.
        classes (None): an optional list of classes to use for zero-shot prediction.

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    model_type = MODEL_TYPES[model_name]

    d = dict(model_path=model_path, classes=classes)

    if model_type == "detection":
        config = fouu.FiftyOneYOLODetectionModelConfig(d)
        return fouu.FiftyOneYOLODetectionModel(config)

    return None


MODEL_URLS = {
    "saurabheights/rf-detr-nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "saurabheights/rf-detr-small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "saurabheights/rf-detr-medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "saurabheights/rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # Better for fine-tuning, not so much for inference. Probably overfitted to coco.
    "saurabheights/rf-detr-base-2": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "saurabheights/rf-detr-large": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
}

MODEL_TYPES = {
    "saurabheights/rf-detr-nano": "detection",
    "saurabheights/rf-detr-small": "detection",
    "saurabheights/rf-detr-medium": "detection",
    "saurabheights/rf-detr-base": "detection",
    "saurabheights/rf-detr-base-2": "detection",
    "saurabheights/rf-detr-large": "detection",
}
