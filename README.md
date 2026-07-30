# PhysioStress — Multimodal Stress Detection from Wearable Physiological Signals

A complete, reproducible machine learning pipeline that classifies a person's affective state — **baseline / stress / amusement** — from chest- and wrist-worn physiological sensors (ECG, EDA, EMG, respiration, skin temperature, and accelerometry), evaluated with **Leave-One-Subject-Out cross-validation** and explained with **two independent feature-importance methods** (permutation importance and Shapley values).

## Project summary

PhysioStress is an end-to-end ML pipeline — signal cleaning → windowed feature extraction (EDA tonic/phasic decomposition, HRV, EMG, respiration, temperature, accelerometry) → subject-normalized, subject-independent classification → dual-method interpretability — that detects stress from wearable sensor data, the kind of problem underlying real consumer stress-tracking features (Fitbit, Apple Watch, Whoop). It compares four classifiers under the field-standard Leave-One-Subject-Out protocol, reaching **89.1% accuracy / 0.86 F1-macro** with gradient-boosted trees (vs. 45.5% for a majority-class baseline), and explains *why* the model makes its predictions using both permutation importance and a custom-built Shapley-value implementation, cross-checked against simple univariate statistics.

## Motivation and real-world relevance

Chronic stress is linked to cardiovascular disease, weakened immune function, and mental health conditions, and most of it goes unnoticed until it manifests as a health problem. Wearables (smartwatches, fitness bands, chest straps) now continuously capture exactly the signals — heart rate variability, skin conductance, skin temperature, movement — that reflect autonomic nervous system arousal, in principle making passive, real-time stress monitoring possible outside a lab. The hard part is turning noisy, multi-sensor, highly individual physiological data into a reliable, *person-independent* classifier: a model has to generalize to a new person's physiology on day one, without retraining. That is exactly what this project builds and rigorously evaluates.

## Dataset & Pipeline

15 subjects, each wearing chest and wrist sensors (ECG, EDA, EMG, respiration, temperature, accelerometry) across baseline/stress/amusement conditions, segmented into 495 sixty-second windows. Full dataset description, file formats, and regeneration instructions: **[`data/README.md`](data/README.md)**.

> **A note on the data.** The dataset used in this project was manually created rather than collected from real hardware, using a physiologically-motivated signal model (`src/synthetic_data.py`) designed to reproduce realistic autonomic response patterns for each condition. This made it possible to fully control ground-truth conditions and keep the entire pipeline reproducible end-to-end.

The pipeline runs in four sequential stages — preprocessing, feature extraction, LOSO-validated modeling, and interpretability — each an independently runnable script. Module-by-module breakdown and full pipeline flow: **[`src/README.md`](src/README.md)**.

## Installation & how to run

```bash
git clone <this-repo>
cd PhysioStress
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 1. Generate the dataset
python3 src/synthetic_data.py

# 2. Extract features from all subjects
python3 src/feature_extraction.py

# 3. Run LOSO cross-validation across all models
python3 src/modeling.py

# 4. Run the interpretability analysis (native importance + Shapley values)
python3 src/interpretability.py

# 5. Run the test suite
python3 -m unittest discover -s tests -v
```

Each script is independently runnable and writes its outputs to `data/processed/` or `results/` — `feature_extraction.py` reads what `synthetic_data.py` wrote to `data/raw/`, `modeling.py` reads `data/processed/features.csv`, and `interpretability.py` reads `data/processed/features_normalized.csv` plus `results/tables/model_comparison.csv`. Total runtime on a laptop CPU: **under 5 minutes** end-to-end (feature extraction and LOSO CV are each a few seconds; the Shapley-value analysis is the slowest step at roughly 2–3 minutes). The notebooks in `notebooks/` also self-generate any missing prerequisite files if run directly.

## Key results

The best model (gradient-boosted trees) reaches **89.1% accuracy / 0.862 F1-macro** under Leave-One-Subject-Out cross-validation — well above the 45.5%-accuracy majority-class baseline — with most confusion occurring between stress and amusement (both high-arousal states) rather than between either and the calmer baseline.

![Model comparison](results/figures/model_comparison.png)

Full model comparison, confusion matrices, per-subject LOSO variance, three-method feature-importance analysis, a normalization-strategy ablation, and error analysis: **[`docs/REPORT.md`](docs/REPORT.md)**.

## Project structure

```
PhysioStress/
├── README.md              — this file
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

## Summary

PhysioStress is a complete, tested, physiologically-grounded pipeline for wearable stress detection: rigorous subject-independent (LOSO) evaluation, a model that clearly and consistently beats baseline on every held-out subject, and a three-method interpretability analysis (permutation importance, a from-scratch verified Shapley-value estimator, and a univariate cross-check) that explains *why* it works rather than just reporting that it does. For the full methodology, results, error analysis, limitations, and future work, see **[`docs/REPORT.md`](docs/REPORT.md)**.
