# Hybrid K-Means & Logistic Regression Classifier

> A machine learning pipeline demonstrating model performance enhancement through localized classification. By combining unsupervised **K-Means Clustering** with supervised **Logistic Regression** using `scikit-learn`, this hybrid approach captures non-linear sub-population behaviors, achieving a **+7.03% boost in ROC-AUC** over standard linear classification models.

**Author:** Eli Carico  
**Course:** NCSU DSA 412 — Final Project (April 2026)

---

## Technical Highlights

- **Unsupervised Sub-Space Partitioning:** Utilized K-Means clustering ($k=6$) to partition continuous high-dimensional feature spaces into localized sub-regions.
- **Localized Supervised Training:** Trained independent `scikit-learn` Logistic Regression estimators across individual clusters to model localized decision boundaries.
- **Optimal Hyperparameter Selection:** Determined cluster count ($k$) programmatically using the Elbow Method via `yellowbrick.cluster.KElbowVisualizer`.
- **Data Leakage Prevention:** Rigorously fitted feature scalers (`StandardScaler`) and clustering models on training partitions only prior to test inference.

---

## Dataset Summary

- **Source:** [UCI Machine Learning Repository - MAGIC Gamma Telescope Dataset](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)
- **Observations:** 19,020 (12,332 Gamma `g`, 6,688 Hadron `h`)
- **Features:** 10 continuous geometric features derived from Cherenkov light ellipsoids
- **Preprocessing:** Target binary mapping (`g=1`, `h=0`), 80/20 stratified train/test split, and standard feature scaling.

---

## Methodology

1. **Standardization:** Applied `StandardScaler` to prevent high-magnitude features from dominating distance metrics.
2. **Optimal Clustering ($k=6$):** Identified using the Elbow Method (`KElbowVisualizer`) on distortion scores.
3. **Baseline Model:** Global Logistic Regression trained across the full training dataset.
4. **Hybrid Model:** Data partitioned into $k=6$ K-Means clusters; independent Logistic Regression classifiers trained on each cluster segment.

---

## Key Results

| Model Architecture | Partitioning Strategy | Local Estimators | Test ROC-AUC | Metrics Boost |
| :--- | :--- | :--- | :---: | :---: |
| **Baseline Model** | None (Global) | Single Logistic Regression | **0.8432** | Reference |
| **Hybrid Pipeline** | K-Means ($k=6$) | Localized Logistic Regressions | **0.9025** | **+0.0593 (+7.03%)** |

---

## Key Takeaways & Impact

Standard linear classifiers often fail when global feature boundaries display complex non-linear relationships. By clustering data points into homogeneous localized regions, simple linear classifiers can fit sub-space decision boundaries effectively—granting non-linear classification capabilities with minimal computational overhead.
