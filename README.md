# Network Intrusion Detection System : Comparative Evaluation of Deep Learning & Classical Machine Learning

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models](#models)
- [Results Summary](#results-summary)
  - [Binary Classification](#binary-classification)
  - [Multiclass Classification](#multiclass-classification)
- [Hardware & Environment](#hardware--environment)
- [How to Run](#how-to-run)
- [Citation](#citation)

---

## Overview

This repository contains the full implementation of a comparative study evaluating **four model families** on **two benchmark datasets** under **binary and multiclass** network intrusion detection tasks — for a total of **8 trained models**.

The key contributions of this work are:

- A **leak-free preprocessing pipeline** (QuantileTransformer + SMOTE applied strictly within each training fold)
- **5-fold stratified cross-validation** for deep learning models with Youden's J optimal threshold selection
- **Focal Loss** (γ=2.0, α=0.25) applied consistently across all deep learning models
- Comprehensive evaluation including accuracy, F1, ROC-AUC, FPR, detection time, and **throughput** — a metric rarely reported in NIDS literature
- A rigorous classical baseline (RF, XGBoost) showing that deep learning proposals without strong classical comparison are methodologically insufficient

---

## Repository Structure

```
.
├── datasets/
│   └── README_datasets.md          # Instructions to download CIC-IDS2017 & UNSW-NB15
│
├── preprocessing/
│   └── pipeline.py                 # Shared preprocessing: QuantileTransformer, SMOTE, TargetEncoder
│
├── models/
│   ├── cnn/
│   │   ├── cnn_cicids2017_binary.ipynb
│   │   ├── cnn_cicids2017_multiclass.ipynb
│   │   ├── cnn_unswnb15_binary.ipynb
│   │   └── cnn_unswnb15_multiclass.ipynb
│   ├── cnn_bilstm/
│   │   ├── cnn_bilstm_cicids2017_binary.ipynb
│   │   ├── cnn_bilstm_cicids2017_multiclass.ipynb
│   │   ├── cnn_bilstm_unswnb15_binary.ipynb
│   │   └── cnn_bilstm_unswnb15_multiclass.ipynb
│   └── classical/
│       ├── rf_xgboost_unswnb15_binary.ipynb
│       └── xgboost_unswnb15_multiclass.ipynb
│
├── results/
│   ├── figures/                    # Confusion matrices, ROC curves, per-fold plots
│   
│
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Records | Features | Classes (Binary) | Classes (Multiclass) |
|---|---|---|---|---|
| **CIC-IDS2017** | 2,830,743 | 78 | Normal / Attack | 15 classes |
| **UNSW-NB15** | 2,540,044 | 49 | Normal / Attack | 10 classes |

- **CIC-IDS2017** — Canadian Institute for Cybersecurity. Contains modern attack scenarios (DDoS, DoS, PortScan, BruteForce, Web Attacks, Bot, Heartbleed, etc.). Extremely imbalanced (up to 191,678:1 ratio for rare classes).
- **UNSW-NB15** — Australian Centre for Cyber Security (ACCS). Contains Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms.

> Datasets are not included in this repository. Download instructions are provided in `datasets/README_datasets.md`.

---

## Preprocessing Pipeline

A **unified, leak-free pipeline** is applied to all models:

1. **Infinity/NaN handling** — Division-by-zero artifacts in CIC-IDS2017 replaced with column mean
2. **Feature scaling** — `QuantileTransformer` (uniform output distribution), fit only on training data
3. **Feature selection** — Top 30 features by Mutual Information score (classical models only)
4. **Target encoding** — `TargetEncoder` fit strictly on training fold (classical models only)
5. **Class imbalance** — `SMOTE` applied **only within each training fold** to prevent data leakage
6. **Loss function** — `Focal Loss` (γ=2.0, α=0.25) for all deep learning models
7. **Threshold selection** — Youden's J statistic (J = TPR − FPR) on validation set for binary tasks

> ⚠️ **Important:** SMOTE is never applied to validation or test sets. This is a critical distinction from many prior works that inflate their results by oversampling before the train/test split.

---

## Models

### Model 1 & 2 — CNN1D (1D Convolutional Neural Network)

| Parameter | CIC-IDS2017 Binary | CIC-IDS2017 Multiclass | UNSW-NB15 Binary | UNSW-NB15 Multiclass |
|---|---|---|---|---|
| Architecture | 3× Conv1D (1→64→128→256) + GAP + 2×FC | same | same | same |
| Output | 2 classes | 15 classes | 2 classes | 10 classes |
| Dropout | 0.3 (conv), 0.4 (FC) | same | same | same |
| Optimizer | Adam (lr=1e-3) | same | same | same |
| Loss | Focal (γ=2, α=0.25) | Focal (γ=2.0) | Focal (γ=2, α=0.25) | Focal (γ=2.0) |
| Epochs / Batch | 100 / 512 | 100 / 512 | 100 / 256 | 50 / 256 |
| Early Stopping | patience=7 | patience=5 | patience=7 | patience=5 |

### Model 3 & 4 — CNN-BiLSTM (Convolutional + Bidirectional LSTM)

| Parameter | CIC-IDS2017 Binary | CIC-IDS2017 Multiclass | UNSW-NB15 Binary | UNSW-NB15 Multiclass |
|---|---|---|---|---|
| Architecture | 2× Conv1D (1→64→128) + BiLSTM (128→256) + 2×FC | same | same | same |
| Output | 2 classes | 15 classes | 2 classes | 10 classes |
| Dropout | 0.3 (conv/LSTM/FC) | same | same | same |
| Optimizer | Adam (lr=1e-3) | same | same | same |
| Loss | Focal (γ=2, α=0.25) | Focal (γ=2.0) | Focal (γ=2, α=0.25) | Focal (γ=2.0) |
| Epochs / Batch | 30 / 64 | 30 / 128 | 100 / 256 | 30 / 128 |
| Early Stopping | patience=5 | patience=5 | patience=7 | patience=5 |

### Model 5 & 6 — Random Forest

- Ensemble of decision trees with bootstrap sampling and feature randomness
- `RandomizedSearchCV` with 5-fold cross-validation for hyperparameter tuning (UNSW-NB15 only)
- Split: 70% train / 10% validation / 20% test (stratified)

### Model 7 & 8 — XGBoost

- Gradient boosted trees with GPU acceleration (NVIDIA L4)
- Custom Focal Loss objective
- SMOTE-resampled training set (651,000 instances for multiclass, ≈65,100 per class)
- Early stopping at iteration 61/315 (multiclass)
- **25× faster training than Random Forest** (203 s vs 5,069 s on binary task)

---

## Results Summary

### Binary Classification

| Model | Dataset | Accuracy | Recall (DR) | F1 | FPR | ROC-AUC | Throughput |
|---|---|---|---|---|---|---|---|
| **CNN** | CIC-IDS2017 | 0.9975 | 0.9987 | **0.9937** | 0.0019 | **0.9999** | **102,397 s/s** |
| CNN-BiLSTM | CIC-IDS2017 | 0.9959 | 0.9981 | 0.9918 | 0.0036 | 0.9999 | 21,828 s/s |
| CNN | UNSW-NB15 | 0.9523 | 0.9407 | 0.9617 | 0.0317 | 0.9942 | 80,055 s/s |
| **CNN-BiLSTM** | UNSW-NB15 | 0.9682 | **0.9594** | 0.9728 | 0.0325 | 0.9977 | 49,853 s/s |
| Random Forest | UNSW-NB15 | **0.9854** | 0.9821 | **0.9885** | — | 0.9989 | — |
| XGBoost | UNSW-NB15 | 0.9852 | 0.9822 | 0.9883 | — | **0.9991** | — |

**Key takeaways:**
- CNN achieves near-perfect discrimination on CIC-IDS2017 (ROC-AUC = 0.9999) with the highest throughput (~102K samples/s), making it well-suited for real-time monitoring.
- CNN-BiLSTM offers +1.87 pp improvement in detection rate on UNSW-NB15 at the cost of 4.7× lower throughput.
- Classical models (RF, XGBoost) **match or exceed deep learning** on UNSW-NB15 binary task without any architectural complexity, confirming the importance of strong baselines.

---

### Multiclass Classification

| Model | Dataset | Classes | Accuracy | Macro F1 | Macro ROC-AUC | Throughput |
|---|---|---|---|---|---|---|
| **CNN** | CIC-IDS2017 | 15 | **0.9893** | **0.7008** | 0.9951 | **104,091 s/s** |
| CNN-BiLSTM | CIC-IDS2017 | 15 | 0.9630 | 0.6713 | **0.9954** | 30,708 s/s |
| CNN | UNSW-NB15 | 10 | 0.7208 | 0.4999 | **0.9645** | 85,428 s/s |
| CNN-BiLSTM | UNSW-NB15 | 10 | 0.7649 | 0.5141 | 0.9611 | 22,069 s/s |
| **XGBoost** | UNSW-NB15 | 10 | **0.8416** | **0.6140** | N/A | — |

**Key takeaways:**
- CNN outperforms CNN-BiLSTM on CIC-IDS2017 (macro F1: 0.701 vs 0.671) with a 3.4× throughput advantage and no training instability.
- XGBoost **dominates all models** on the 10-class UNSW-NB15 task (macro F1: 0.614 vs 0.514 for CNN-BiLSTM), particularly on Reconnaissance (F1=0.831) and Fuzzers (F1=0.801).
- Minority attack classes (Analysis, Backdoor, Worms, SQL Injection) remain a persistent challenge across all architectures, driving macro F1 well below accuracy despite high ROC-AUC values.

---

## Hardware & Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA L4 (23.7 GB VRAM, Ada Lovelace) |
| CUDA | 13.0 (driver 580.82.07) |
| PyTorch | 2.11.0 (stable) |
| XGBoost | GPU-accelerated (`device='cuda'`) |
| Python | 3.10+ |

**Key libraries:**
```
torch
xgboost
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/nids-comparative-evaluation.git
cd nids-comparative-evaluation
```

### 2. Download the datasets

See `datasets/README_datasets.md` for download links and placement instructions.

### 3. Run a notebook

```bash
jupyter notebook models/cnn1d/cnn1d_cicids2017_binary.ipynb
```

Or run all experiments sequentially via the provided script:

```bash
python run_all_experiments.py
```

### 4. Notebook naming convention

Each notebook follows the pattern: `{model}_{dataset}_{task}.ipynb`

| Notebook | Model | Dataset | Task |
|---|---|---|---|
| `cnn1d_cicids2017_binary.ipynb` | CNN1D | CIC-IDS2017 | Binary |
| `cnn1d_cicids2017_multiclass.ipynb` | CNN1D | CIC-IDS2017 | 15-class |
| `cnn1d_unswnb15_binary.ipynb` | CNN1D | UNSW-NB15 | Binary |
| `cnn1d_unswnb15_multiclass.ipynb` | CNN1D | UNSW-NB15 | 10-class |
| `cnn_bilstm_cicids2017_binary.ipynb` | CNN-BiLSTM | CIC-IDS2017 | Binary |
| `cnn_bilstm_cicids2017_multiclass.ipynb` | CNN-BiLSTM | CIC-IDS2017 | 15-class |
| `cnn_bilstm_unswnb15_binary.ipynb` | CNN-BiLSTM | UNSW-NB15 | Binary |
| `cnn_bilstm_unswnb15_multiclass.ipynb` | CNN-BiLSTM | UNSW-NB15 | 10-class |
| `rf_xgboost_unswnb15_binary.ipynb` | RF & XGBoost | UNSW-NB15 | Binary |
| `xgboost_unswnb15_multiclass.ipynb` | XGBoost | UNSW-NB15 | 10-class |

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@article{cheick2025nids,
  title   = {Deep Learning and Classical Machine Learning for Network Intrusion Detection:
             A Comparative Evaluation on CIC-IDS2017 and UNSW-NB15},
  author  = {Cheick Mohamed, Rachid and Tonyalı, Samet},
  journal = {Computers \& Security},
  year    = {2025},
  note    = {Preprint submitted to Elsevier}
}
```

---

## Authors

**Rachid Cheick Mohamed** — Master's student, Dept. of Artificial Intelligence and Intelligent Systems, Gümüşhane University, Türkiye
`cheickmohamedrachid@outlook.com` · ORCID: [0009-0009-6471-5789](https://orcid.org/0009-0009-6471-5789)

**Samet Tonyalı** — Assistant Professor, Dept. of Software Engineering, Gümüşhane University, Türkiye
`samet.tonyali@gumushane.edu.tr` · ORCID: [0000-0001-7799-2771](https://orcid.org/0000-0001-7799-2771)

---

*Gümüşhane University — Department of Artificial Intelligence and Intelligent Systems*
