# Semantic Drone Segmentation with DeepLabv3

This repository contains PyTorch code for training a semantic segmentation model on the [Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset). The model is based on **DeepLabv3-ResNet50** and achieves **\~70% F1 score** on the validation set.

---

## 📂 Dataset

The dataset can be downloaded from Kaggle:

* **Semantic Drone Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)

It contains:

* `original_images/` → RGB drone images
* `label_images_semantic/` → Pixel-level semantic labels
* `class_dict_seg.csv` → Class color mapping

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchvision albumentations opencv-python pandas scikit-learn tqdm
```

---

## 🚀 Training

Run the training script:

```bash
python train.py
```

Key configuration (set in `train.py`):

* Image size: `512x512`
* Batch size: `2`
* Epochs: `30`
* Learning rate: `1e-4`
* Loss: CrossEntropy + Dice Loss
* Optimizer: AdamW

During training, validation predictions will be saved in `val_outputs/epXX/`.

---

## 📊 Results

* Best **Validation F1 Score**: \~**70%**
* Model checkpoint saved as `best_drone_segmentation.pth`

Example output comparison:

```
(val_outputs/ep05/example_comparison.png)
```

Left: Input image | Right: Predicted segmentation

---

## 🧪 Inference

To run inference on new images:

```python
import torch, cv2
from train import preprocess, colorize

# Load model
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
model.classifier = DeepLabHead(2048, NUM_CLASSES)
model.load_state_dict(torch.load("best_drone_segmentation.pth", map_location="cpu"))
model.eval()

# Predict
img = cv2.imread("sample.jpg")[..., ::-1]
input_tensor = preprocess(img).unsqueeze(0)
with torch.no_grad():
    pred = model(input_tensor)["out"][0].argmax(0).numpy()

# Colorize result
colored = colorize(pred)
cv2.imwrite("sample_pred.png", colored[..., ::-1])
```

---

## 📈 Future Work

* Add data augmentations (rotation, crop, blur)
* Try other backbones (ResNet101, MobileNetV3)
* Evaluate mIoU along with F1

---

## 📜 License

This project is released under the MIT License.

Dataset © by [Semantic Drone Dataset authors](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset).
