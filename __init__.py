"""
Roboflow RF-DETR models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import eta.core.web as etaw
import fiftyone as fo
import supervision
import torch
import torchvision.transforms.functional as F
from fiftyone.core.models import Model, SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem
from PIL import Image
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


class RfDetrGetItem(GetItem):
    """Data loader transform for SHARP model. Just passes filepath to predict_all."""

    def __init__(
        self,
        field_mapping: Optional[Dict[str, str]] = None,
        device=None,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(field_mapping=field_mapping)
        self.resolution = resolution
        self.device = device

    @property
    def required_keys(self) -> List[str]:
        return ["filepath"]

    def __call__(self, sample_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        img = Image.open(sample_dict["filepath"])
        img_tensor = F.to_tensor(img)
        # if self.device:  # Cause error due to multiprocessing not using spawn.
        #     img_tensor = img_tensor.to(self.device)
        if self.resolution:
            img_tensor = F.resize(img_tensor, self.resolution)
        img_tensor[img_tensor > 1] = 1
        return img_tensor


class RfDetrModel(Model, SupportsGetItem, TorchModelMixin):
    """A fifty-one wrapper class for the Roboflow RF-DETR model."""

    def __init__(self, model_name, model_path, **kwargs):
        SupportsGetItem.__init__(self)
        self._preprocess = False

        def get_model(model_name, batch_size, optimize):
            model_class = {
                "saurabheights/rf-detr-nano": RFDETRNano,
                "saurabheights/rf-detr-small": RFDETRSmall,
                "saurabheights/rf-detr-medium": RFDETRMedium,
                "saurabheights/rf-detr-large": RFDETRLarge,
            }

            rfdetr_model = model_class[model_name](pretrain_weights=model_path)
            if optimize:
                logging.info(f"Optimizing model: {model_name} for inference.")
                rfdetr_model.optimize_for_inference(
                    compile=True,
                    batch_size=batch_size,
                )
                logging.info(f"Model: {model_name} optimized for inference.")
            return rfdetr_model

        self.threshold = kwargs.get("threshold", 0.5)
        self.batch_size = kwargs.get("batch_size", 1)
        optimize = kwargs.get("optimize", False)
        self._model = get_model(model_name, self.batch_size, optimize)
        self.test_time_augmentation = kwargs.get("test_time_augmentation", False)
        self.tta_threshold = kwargs.get("tta_threshold", 0.5)
        logging.debug(
            f"Loading Rf-Detr model {model_name} with Threshold: {self.threshold}"
        )

    @property
    def media_type(self):
        return "image"

    @property
    def ragged_batches(self) -> bool:
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value: bool):
        self._preprocess = value

    # Methods from SupportsGetItem
    def build_get_item(self, field_mapping: Optional[Dict[str, str]] = None) -> GetItem:
        return RfDetrGetItem(
            field_mapping=field_mapping,
            device=self._model.model.device,
            resolution=(self._model.model.resolution, self._model.model.resolution),
        )

    @property
    def has_collate_fn(self) -> bool:
        return True

    @property
    def collate_fn(self):
        return lambda batch: batch

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit - clear GPU memory cache."""
        # Clear cache based on device type (don't move model to CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

    def predict(self, arg):
        batch_detections: supervision.Detections | List[supervision.Detections] = (
            self._model.predict(arg, threshold=self.threshold)
        )

        if isinstance(batch_detections, supervision.Detections):
            batch_detections = [batch_detections]

        # Convert detections to FiftyOne format
        c, h, w = arg[0].shape
        results = []
        for detections in batch_detections:
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
            results.append(fo.Detections(detections=fo_detections))
        return results

    def predict_all(self, batch):
        return self.predict(batch)


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
    "saurabheights/rf-detr-large-2026": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
}

MODEL_TYPES = {
    "saurabheights/rf-detr-nano": "detection",
    "saurabheights/rf-detr-small": "detection",
    "saurabheights/rf-detr-medium": "detection",
    "saurabheights/rf-detr-base": "detection",
    "saurabheights/rf-detr-base-2": "detection",
    "saurabheights/rf-detr-large": "detection",
}
