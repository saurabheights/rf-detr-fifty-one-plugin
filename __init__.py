"""
Roboflow RF-DETR models.
"""

import logging

import eta.core.web as etaw
import fiftyone as fo
import supervision
from fiftyone.core.models import Model
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


class RfDetrModel(Model):
    """A fifty-one wrapper class for the Roboflow RF-DETR model."""

    def __init__(self, model_name, model_path, **kwargs):
        super().__init__()

        def get_model(model_name):
            model_class = {
                "saurabheights/rf-detr-nano": RFDETRNano,
                "saurabheights/rf-detr-small": RFDETRSmall,
                "saurabheights/rf-detr-medium": RFDETRMedium,
                "saurabheights/rf-detr-large": RFDETRLarge,
            }

            rfdetr_model = model_class[model_name](pretrain_weights=model_path)
            logging.info(f"Optimizing model: {model_name} for inference.")
            rfdetr_model.optimize_for_inference()
            logging.info(f"Model: {model_name} optimized for inference.")
            return rfdetr_model

        self._model = get_model(model_name)
        self.threshold = kwargs.get("threshold", 0.5)
        logging.debug(
            f"Loading Rf-Detr model {model_name} with Threshold: {self.threshold}"
        )

    @property
    def media_type(self):
        return "image"

    def predict(self, arg):
        h, w, c = arg.shape

        detections: supervision.Detections = self._model.predict(
            arg, threshold=self.threshold
        )

        # Convert detections to FiftyOne format
        fo_detections: fo.Detections = []
        for bbox_xyxy, class_id, confidence in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            x1, y1, x2, y2 = bbox_xyxy
            normalized_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            fo_detections.append(
                fo.Detection(
                    label=self._model.class_names[class_id.item()],
                    bounding_box=normalized_box,
                    confidence=confidence.item(),
                )
            )
        return fo.Detections(detections=fo_detections)


def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download.
        model_path: the path to download the model to.
    """
    url = MODEL_URLS[model_name]
    etaw.download_file(url, path=model_path)


def load_model(model_name, model_path, classes=None, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load.
        model_path: the path to download the model to.
        classes (None): an optional list of classes to use for zero-shot prediction.

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    model_type = MODEL_TYPES[model_name]
    if model_type == "detection":
        return RfDetrModel(model_name, model_path, **kwargs)

    return None


MODEL_URLS = {
    "saurabheights/rf-detr-nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "saurabheights/rf-detr-small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "saurabheights/rf-detr-medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "saurabheights/rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
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
