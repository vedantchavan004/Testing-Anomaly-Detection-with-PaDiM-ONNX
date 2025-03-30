# PaDiM Anomaly Detection with ONNX and MobileNetV3

A lightweight and interpretable anomaly detection pipeline based on **PaDiM (Patch Distribution Modeling)** using a **MobileNetV3-Small** backbone. This project demonstrates how to detect anomalies in industrial images (e.g. transistor defects) and export the model for efficient inference using **ONNX Runtime**.

---

## 📁 Project Structure

```
/content/transistor_padim/
├── mobilenet_v3_small_with_intermediates.onnx   # ONNX-exported feature extractor
├── padim_params.npz                             # Mean, inverse covariance & config
├── visualizations/                              # Output result images (inference)
```

---

## 🚀 How it Works

### Training & Export
- **Loads MobileNetV3** pretrained on ImageNet
- **Extracts intermediate features** from selected layers (e.g., `features.1`, `features.3`, `features.8`)
- **Combines and reduces feature dimensions** to a manageable vector size (100D)
- **Computes statistical representations** (mean vector and inverse covariance matrix) for each spatial pixel location
- **Saves**:
  - The trained feature extractor in ONNX format
  - The statistical parameters as `.npz`

### Inference with ONNX
- Loads the ONNX model and PaDiM parameters
- Preprocesses each test image (resize, normalize)
- Runs it through the ONNX model to extract features
- Resizes, reduces, and concatenates feature maps
- Computes **Mahalanobis distance** between test features and training distribution
- Generates anomaly score maps, binary masks, and visualizations

---

## 📸 Sample Output

Each image in the test set generates a visualization like:
- Original Image
- Anomaly Heatmap
- Binary Mask
- Overlay (Image + Heatmap)

Stored in: `/content/transistor_padim/visualizations`

### 🔍 Example Results

| Original → Heatmap → Mask → Overlay |
|:--:|
| <img src="images/result_007.png" width="800"/> |
| <img src="images/result_008.png" width="800"/> |

---

## 📐 Mahalanobis Distance: The Core Idea

PaDiM uses the **Mahalanobis distance** to measure how much a feature vector at a given location differs from the normal distribution (learned during training).

> Unlike Euclidean distance, Mahalanobis accounts for feature correlations and scales.

Mathematically:
```
D(x) = sqrt((x - μ)^T · Σ^(-1) · (x - μ))
```
Where:
- `x` = feature vector from test image
- `μ` = mean vector from training data
- `Σ^(-1)` = inverse of the covariance matrix

This distance tells us how “strange” a point is. Larger = more anomalous.

If a pixel's Mahalanobis distance exceeds a certain threshold (e.g., 98th percentile), it is considered **anomalous**.

---

## 🛠️ Use Cases
- Visual Quality Inspection in Electronics Manufacturing
- Fault Detection for Mechanical Components
- Surface Defect Detection in Industrial Products

---

## 💬 Credits
Built using PyTorch, ONNX, and PaDiM principles from the original CVPR 2021 paper:

"Student-teacher Feature Pyramid Matching for Unsupervised Anomaly Detection" by Defard et al.

Dataset: MVTec AD Dataset
---


