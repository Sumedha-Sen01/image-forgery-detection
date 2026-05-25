# Image Forgery Detection using Machine Learning and Image Processing
## Overview
This project focuses on detecting forged or tampered images using advanced Image Processing and Machine Learning techniques. The system extracts multiple image features, performs preprocessing and normalization, applies dimensionality reduction using PCA, and evaluates various machine learning models for classification.

The project combines both spatial-domain and frequency-domain feature extraction methods to improve forgery detection accuracy and robustness.

---

## Objectives
- Detect forged/tampered images effectively
- Compare different machine learning models
- Improve detection performance using feature fusion
- Reduce dimensionality using PCA
- Analyze model performance before and after PCA

---

## Datasets Used
The project uses:
- CASIA Image Tampering Detection Dataset
- MICC Image Forgery Dataset

> Note: Large datasets are not uploaded to GitHub due to size limitations.

Dataset Link:
Add your Google Drive/Kaggle dataset link here.

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- PyWavelets
- Matplotlib
- Jupyter Notebook

---

#Project Workflow

## 1. SIFT Feature Extraction
Scale-Invariant Feature Transform (SIFT) is used to extract local image descriptors from images.

### Process:
- Keypoint detection
- Descriptor extraction
- Descriptor sampling
- KMeans clustering
- Histogram generation

### Outputs:
- CASIA_SIFT.csv
- MICC_SIFT.csv

---

## 2. Wavelet + DCT Feature Extraction
Frequency-domain features are extracted using:
- Wavelet Transform
- Discrete Cosine Transform (DCT)

These features help identify inconsistencies introduced during image tampering.

### Outputs:
- CASIA_Wavelet.csv
- MICC_Wavelet.csv

---

## 3. Feature Fusion
The extracted SIFT and Wavelet/DCT features are combined into a unified feature representation.

### Benefits:
- Improved learning capability
- Better classification accuracy
- Robust feature representation

### Outputs:
- CASIA_Fusion.csv
- MICC_Fusion.csv

---

## 4. Z-Score Normalization
Feature values are standardized using Z-score normalization to ensure balanced model training.

### Outputs:
- CASIA_Normalized.csv
- MICC_Normalized.csv
- Saved scaler models (.pkl)

---

## 5. Pre-PCA Model Training and Evaluation
Several machine learning models are trained before PCA dimensionality reduction.

### Models Used:
- Random Forest
- XGBoost
- Gradient Boosting
- Artificial Neural Network (ANN)

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation Accuracy

---

## 6. PCA Dimensionality Reduction
Principal Component Analysis (PCA) is applied to:
- Reduce feature dimensions
- Remove redundancy
- Improve computational efficiency

### Outputs:
- PCA-transformed datasets
- Saved PCA model files

---

## 7. Post-PCA Model Evaluation
The same machine learning models are re-evaluated after PCA.

### Purpose:
- Compare performance before and after dimensionality reduction
- Analyze efficiency and accuracy trade-offs

---

## 8. Final Performance Comparison
The final stage compares:
- Pre-PCA model performance
- Post-PCA model performance
- Best-performing classification model

---

# Features
- Multi-stage feature extraction
- Hybrid feature fusion
- Frequency and spatial domain analysis
- PCA-based dimensionality reduction
- Multiple machine learning model comparison
- Robust preprocessing pipeline
- Automated evaluation metrics
