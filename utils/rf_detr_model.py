import logging

import supervision
from fiftyone.core.models import Model
import fiftyone as fo
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


class RfDetrModel(Model):
    """A model that uses the Roboflow RF-DETR model."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        def get_model(model_name):
            model_class = {
                "saurabheights/rf-detr-nano": RFDETRNano,
                "saurabheights/rf-detr-small": RFDETRSmall,
                "saurabheights/rf-detr-medium": RFDETRMedium,
                "saurabheights/rf-detr-base": RFDETRBase,
                "saurabheights/rf-detr-large": RFDETRLarge,
            }
            rfdetr_model = model_class[model_name]()
            logging.info(f"Optimizing model: {model_name} for inference.")
            rfdetr_model.optimize_for_inference()
            logging.info(f"Model: {model_name} optimized for inference.")
            return rfdetr_model

        self._model = get_model("rfdetr-m")

    @property
    def media_type(self):
        return "image"

    def predict(self, arg):
        h, w, c = arg.shape
        detections: supervision.Detections = self._model.predict(arg, threshold=0.5)

        # Convert detections to FiftyOne format
        fo_detections: fo.Detections = []
        for bbox_xyxy, class_id, confidence in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            x1, y1, x2, y2 = bbox_xyxy
            normalized_box = [x1 / w, y1 / h, (x2-x1) / w, (y2-y1) / h]
            fo_detections.append(
                fo.Detection(
                    label=self._model.class_names[class_id.item()],
                    bounding_box=normalized_box,
                    confidence=confidence.item()
                )
            )
        return fo.Detections(detections=fo_detections)
