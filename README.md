# Fifty-One Plugin for RF-Detr Models

Wrapper for various [RF-DETR models](https://github.com/roboflow/rf-detr/) for the
FiftyOne Model Zoo.

## Example usage

```py
import fiftyone as fo
import fiftyone.zoo as foz

# Load any sample dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=50,
    shuffle=True,
)

# Register model
foz.register_zoo_model_source("https://github.com/saurabheights/rf-detr-fifty-one-plugin")

# Load the medium-size RF-Detr model and save inference results in rf-detr-m field.
model = foz.load_zoo_model("saurabheights/rf-detr-medium")
dataset.apply_model(model, label_field="rf-detr-m")
```
