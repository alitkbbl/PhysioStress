# PhysioStress — Multimodal Stress Detection from Wearable Physiological Signals

> **An end-to-end, reproducible machine learning pipeline for person-independent affective state classification using multimodal physiological sensors.**

<!-- Badges Header -->
<div align="left">
  
![Task](https://img.shields.io/badge/Task-Stress%20Detection-critical)
![Signals](https://img.shields.io/badge/Signals-Multimodal-8A2BE2)
![Explainability](https://img.shields.io/badge/Explainability-Permutation%20%2B%20Shapley-teal)

</div>

---

## 📌 Project Summary

**PhysioStress** is an end-to-end Machine Learning pipeline that classifies human affective states — **Baseline**, **Stress**, and **Amusement** — from physiological signals captured by chest- and wrist-worn sensors (ECG, EDA, EMG, respiration, skin temperature, and accelerometry).

Designed to mimic real-world consumer stress-tracking features (found in devices like Fitbit, Apple Watch, and Whoop), the pipeline bridges raw sensor processing and actionable interpretability across four core stages:

1.  **Signal Cleaning & Preprocessing:** Filtering noise and artifacts from raw multi-sensor streams.
2.  **Windowed Feature Extraction:** Extracting EDA tonic/phasic components, HRV parameters, EMG power, respiration rates, temperature trends, and accelerometry dynamics.
3.  **Subject-Normalized Classification:** Evaluating 4 classifiers under strict **Leave-One-Subject-Out (LOSO)** cross-validation to guarantee zero data leakage between subjects.
4.  **Dual-Method Interpretability:** Explaining model predictions using both Permutation Importance and a custom Shapley-value implementation, cross-checked against univariate statistics.

---

## 💡 Motivation & Real-World Relevance

Chronic stress is strongly linked to cardiovascular diseases, immune dysfunction, and mental health challenges. However, stress often goes unmanaged until noticeable clinical symptoms appear.

Modern wearables continuously record biological indicators driven by the **Autonomic Nervous System (ANS)**:
*  **Heart Rate Variability (HRV)** — Reflects sympathetic / parasympathetic tone balance.
*  **Electrodermal Activity (EDA)** — Directly indexes sympathetic arousal via sweat gland activation.
*  **Skin Temperature & Respiration** — Capture physiological relaxation vs. fight-or-flight responses.
*  **Accelerometry** — Controls for physical movement artifacts versus true emotional stress.

### The Core ML Challenge:
> **How do we build a model that generalizes to a new user on Day 1 without retraining?**
>
> Physiological baselines vary dramatically between individuals. The primary hurdle in wearable health technology is creating a **person-independent** classifier. PhysioStress addresses this challenge through subject-wise normalization and rigorous zero-leakage cross-validation.

---

## 🔬 Dataset & Pipeline Overview

The dataset covers **15 subjects** monitored across chest and wrist devices during Baseline, Stress, and Amusement protocols, yielding **495 sixty-second windowed samples**.

> [!NOTE]
> **Synthetic Data Notice:** To ensure full end-to-end reproducibility without external hardware constraints, the dataset in this repository was generated using a physiologically motivated signal model (`src/synthetic_data.py`). This script reproduces realistic autonomic response patterns for each affective condition while enabling controlled ground-truth evaluation.

* 📁 **Data Architecture:** Detailed dataset structure and regeneration steps can be found in **[`data/README.md`](data/README.md)**.
* ⚙️ **Modular Execution Pipeline:** The four sequential stages (preprocessing, extraction, LOSO training, XAI) run as independent modules. Full architectural workflow is documented in **[`src/README.md`](src/README.md)**.

---

## 🚀 Getting Started

Follow these steps to set up the environment and reproduce the full pipeline—from synthetic signal generation to model interpretability.

### ⚙️ Installation & Environment Setup

Clone the repository and install the dependencies. Using a virtual environment is highly recommended to keep your global Python setup clean.
```bash
# Clone the repository
git clone https://github.com/alitkbbl/PhysioStress.git
cd PhysioStress

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

```

### ✅ Running the Test Suite

We maintain a suite of unit tests to ensure pipeline reliability, signal processing logic, and data integrity. Before submitting changes or after cloning, verify the project integrity:
```bash
# Run all tests in the 'tests/' directory with verbose output
python3 -m unittest discover -s tests -v
```


---

## Key results

The best model (gradient-boosted trees) reaches **89.1% accuracy / 0.862 macro-F1** under Leave-One-Subject-Out (LOSO) cross-validation — well above the **45.5%** majority-class baseline. Most errors come from **stress vs. amusement** confusion (both high-arousal states), rather than from either vs. the calmer baseline.

![Model comparison](results/figures/model_comparison.png)

### 📖 Full technical report (bilingual):
- [ReportEN.pdf](./ReportEN.pdf) – English version
- [ReportFA.pdf](./ReportFA.pdf) – Persian (Farsi) version

Includes full model comparison, confusion matrices, per-subject LOSO variance, feature-importance analysis (SHAP + permutation + univariate), normalization ablation, and error analysis.

---

## Project structure

```
PhysioStress/
├── README.md             
├── requirements.txt
├── data/
│   ├── README.md            — dataset details
│   ├── raw/                — per-subject .pkl files
│   └── processed/          — extracted feature tables (features.csv, features_normalized.csv)
├── src/
│   ├── README.md             — pipeline/module details
│   ├── utils.py             — shared config: paths, sampling rates, label maps, windowing
│   ├── synthetic_data.py    — physiological signal generator
│   ├── preprocessing.py     — loading, filtering, segmentation, windowing
│   ├── feature_extraction.py— per-window feature computation (EDA/HRV/EMG/RESP/TEMP/ACC)
│   ├── modeling.py          — per-subject normalization, model zoo, LOSO CV
│   ├── interpretability.py  — permutation importance, custom Shapley values, univariate ANOVA
│   └── plots.py             — shared matplotlib/seaborn figure helpers
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_results_walkthrough.ipynb
│   └── 03_loso_training_validation.ipynb  — independent, explicit fold-by-fold LOSO
│                                             train/eval walkthrough (see docs/REPORT.md
│                                             Section 4.4); saves into the same results/
├── results/
│   ├── figures/             — all generated plots (PNG)
│   ├── tables/               — all generated metrics/importance tables (CSV)
│   └── models/                — saved trained models (pickle + joblib)
├── docs/
│   └── REPORT.md             — full technical report
└── tests/                     — unit tests (unittest, run with `python -m unittest discover`)
```
---

## Summary

PhysioStress is a complete, tested, physiologically-grounded pipeline for wearable stress detection: rigorous subject-independent (LOSO) evaluation, a model that clearly and consistently beats baseline on every held-out subject, and a three-method interpretability analysis (permutation importance, a from-scratch verified Shapley-value estimator, and a univariate cross-check) that explains *why* it works rather than just reporting that it does. For the full methodology, results, error analysis, limitations, and future work, see **[`docs/README.md`](docs/README.md)**.
