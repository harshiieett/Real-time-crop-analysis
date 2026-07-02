# 🌿 Real-Time Crop Disease Diagnosis using Deep Learning

> Transfer learning approach for multi-class plant disease classification across 38 disease categories, trained on 70K+ real-world crop images.

---

## 📌 Overview

Agricultural crop diseases cause significant yield losses globally, often going undetected until visible damage is severe. This project builds an automated deep learning pipeline that classifies plant leaf images into one of **38 health/disease categories** in real time, enabling early diagnosis at scale.

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~96% |
| Training Accuracy (at stop) | 98.5% |
| Training Images | 70,295 |
| Validation Images | 17,572 |
| Disease Classes | 38 |
| Epochs to Convergence | 6 |

> Training was halted via a custom early-stopping callback once training accuracy crossed 98.5%. Backbone weights (ResNet101V2) were kept frozen throughout — no fine-tuning was applied.

---

## 🏗️ Architecture

```
ResNet101V2 (frozen, pretrained on ImageNet)
        ↓
  Dropout (0.2)
        ↓
     Flatten
        ↓
  BatchNormalization
        ↓
   Dense(1024) → BatchNorm → ReLU → Dropout(0.5)
        ↓
   Dense(512)  → BatchNorm → ReLU → Dropout(0.5)
        ↓
   Dense(38, softmax)
```

- **Backbone**: ResNet101V2 — chosen for its depth (101 layers) and residual connections, which help preserve gradient flow and feature reuse across deep networks.
- **Custom Head**: Two-stage dense compression with BatchNorm to stabilise training and Dropout to prevent overfitting on the domain-specific classes.
- **Optimizer**: Adam (lr = 0.001), Categorical Crossentropy loss.

---

## 📁 Dataset

**[New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)** — Kaggle

- 38 classes covering healthy and diseased states across 14 crop types
- Crops include: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- Images resized to 224×224 for ResNet101V2 compatibility

> Note: The dataset was loaded with only pixel rescaling (÷255). Data augmentation (rotation, zoom, flip) was explored but not applied in this run — a direction for future improvement.

---

## ⚙️ Setup & Usage

```bash
git clone https://github.com/harshiieett/Real-time-crop-analysis
cd Real-time-crop-analysis
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-image pillow
```

Open `Real Time Crop Analysis.ipynb` in Jupyter Notebook or Kaggle and update the dataset path:

```python
path = '/your/path/to/New Plant Diseases Dataset(Augmented)'
```

Run all cells sequentially. The model is saved as `PlantDiseasePrediction.h5` for reuse.

**Predicting a single image:**
```python
output('/path/to/leaf_image.jpg', 'ActualClass.jpg')
```

**Predicting a folder of images:**
```python
# Enter folder path when prompted
# Predictions and actual labels printed for each image
```

---

## 🔍 Limitations & Future Work

- **Data augmentation disabled** — enabling rotation, zoom, and horizontal flip could improve robustness to real-field image variation
- **Backbone frozen** — fine-tuning the top layers of ResNet101V2 may push validation accuracy further
- **Controlled dataset** — real-world deployment would require testing on field-captured images with variable lighting, angles, and backgrounds
- **38 classes only** — coverage can be extended to additional crops and diseases

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
