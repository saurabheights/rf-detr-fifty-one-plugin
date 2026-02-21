import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.rf_detr_model import RfDetrModel


def read_rgb_image(path):
    """Utility function that loads an image as an RGB numpy array."""
    return np.asarray(Image.open(path).convert("RGB"))


# Load a `Model` instance that processes images
model = RfDetrModel()

# Load a FiftyOne dataset
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    splits=["validation"],
    label_types=["detections"],
    max_samples=200,
    classes=["Human body"],
)

# A sample field in which to store the predictions
label_field = "saurabheights/rf-detr-nano"
dataset.delete_labels(fields=label_field)

# Perform prediction on all images in the dataset
with model:
    for sample in tqdm(dataset):
        # Load image
        img = read_rgb_image(sample.filepath)

        # Perform prediction
        labels = model.predict(img)

        # Save labels
        sample.add_labels(labels, label_field=label_field)
        sample.save()

session = fo.launch_app(dataset=dataset)
session.wait()
