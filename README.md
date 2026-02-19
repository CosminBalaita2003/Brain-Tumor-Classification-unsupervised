# Brain Tumor Classification – Unsupervised Learning Project

## Overview

This project explores **unsupervised learning techniques** for clustering brain tumor MRI images.  
The goal is to group images into meaningful clusters **without using labels during training**, and then evaluate how well the discovered clusters match the true tumor classes.

The project compares:

-  **Two clustering algorithms**
  - DBSCAN
  - Fuzzy C-Means (FCM)

-  **Two feature representations**
  - HOG (Histogram of Oriented Gradients)
  - Handcrafted statistical & texture features

-  **Baselines**
  - Random baseline
  - Supervised baseline (Linear SVM)

---

## Dataset

The dataset consists of MRI brain images organized into 4 classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The dataset is split into:

- **Train** – used for fitting scaler and PCA
- **Validation** – used for hyperparameter tuning
- **Test** – used only for final evaluation

---

##  Pipeline Overview

The full processing pipeline is:

Image → Feature Extraction → Standardization → PCA (Dimensionality Reduction) → Clustering → Mapping to True Labels → Evaluation Metrics


---

## Feature Extraction

### HOG Features

HOG captures edge and shape information using image gradients.

Steps:
- Compute gradients (Gx, Gy)
- Compute magnitude and orientation
- Build orientation histograms
- Normalize per block

Produces a high-dimensional feature vector.

---

### Handcrafted Features

Includes:

- Statistical features (mean, std, skewness, kurtosis)
- Intensity histograms
- LBP (Local Binary Patterns)
- GLCM texture features (contrast, energy, homogeneity)

Produces a compact and interpretable feature vector.

---

## Dimensionality Reduction – PCA

Principal Component Analysis (PCA):

- Reduces dimensionality
- Keeps maximum variance directions
- Removes noise
- Improves clustering stability

Number of components is selected via grid search.

---

## Clustering Methods

###  DBSCAN

Density-based clustering.

Hyperparameters:
- `eps` – neighborhood radius
- `min_samples` – minimum neighbors to form a dense region
- `metric` – distance (euclidean / cosine)

Advantages:
- Detects noise
- Finds arbitrarily shaped clusters

---

###  Fuzzy C-Means (FCM)

Soft clustering algorithm.

Each sample has a membership value in each cluster.

Hyperparameters:
- `m` – fuzziness coefficient
- `metric`
- PCA dimension

---

##  Hyperparameter Tuning

Grid search is performed on the **validation set**.

For each PCA dimension and clustering configuration:

- Train clustering
- Compute validation metrics
- Select best parameters
---

## Evaluation Metrics

Clustering is evaluated using:

- **ARI (Adjusted Rand Index)**
- **NMI (Normalized Mutual Information)**
- **Hungarian Accuracy**

Hungarian algorithm is used to optimally map cluster labels to true class labels.

---

## Plots Generated

- Validation ARI vs PCA dimension
- Validation NMI vs PCA dimension
- Selection score vs PCA dimension
- ARI vs fuzziness parameter (FCM)
- Macro-F1 vs C (SVM baseline)
- Confusion matrix (supervised baseline)

These plots help understand how hyperparameters influence clustering performance.

---

##  Baselines

### Random Baseline
Assigns random cluster labels.
Used to verify clustering performance is above chance.

### Supervised Baseline (Linear SVM)
Trained using labels.
Provides an upper-bound reference for feature quality.

---

## Project Structure
- prepare_dataset.py
- feature_extraction_handcrafted.py
- feature_extraction_hog.py
- main_dbscan.py
- main_fcm.py
- baseline_random.py
- baseline_supervised.py
- outputs/


---

## Conclusion

The project demonstrates:

- The impact of feature representation on clustering quality
- The importance of dimensionality reduction
- The difference between density-based and fuzzy clustering
- How unsupervised learning compares to supervised baselines

It provides both quantitative evaluation and qualitative cluster interpretation.
