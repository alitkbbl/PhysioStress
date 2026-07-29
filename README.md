# PhysioStress — Multimodal Stress Detection from Wearable Physiological Signals

A complete, reproducible machine learning pipeline that classifies a person's affective state — **baseline / stress / amusement** — from chest- and wrist-worn physiological sensors (ECG, EDA, EMG, respiration, skin temperature, and accelerometry), evaluated with **Leave-One-Subject-Out cross-validation** and explained with **two independent feature-importance methods** (permutation importance and Shapley values).

## Project summary

PhysioStress is an end-to-end ML pipeline — signal cleaning → windowed feature extraction (EDA tonic/phasic decomposition, HRV, EMG, respiration, temperature, accelerometry) → subject-normalized, subject-independent classification → dual-method interpretability — that detects stress from wearable sensor data, the kind of problem underlying real consumer stress-tracking features (Fitbit, Apple Watch, Whoop). It compares four classifiers under the field-standard Leave-One-Subject-Out protocol, reaching **89.1% accuracy / 0.86 F1-macro** with gradient-boosted trees (vs. 45.5% for a majority-class baseline), and explains *why* the model makes its predictions using both permutation importance and a custom-built Shapley-value implementation, cross-checked against simple univariate statistics.

## Motivation and real-world relevance

Chronic stress is linked to cardiovascular disease, weakened immune function, and mental health conditions, and most of it goes unnoticed until it manifests as a health problem. Wearables (smartwatches, fitness bands, chest straps) now continuously capture exactly the signals — heart rate variability, skin conductance, skin temperature, movement — that reflect autonomic nervous system arousal, in principle making passive, real-time stress monitoring possible outside a lab. The hard part is turning noisy, multi-sensor, highly individual physiological data into a reliable, *person-independent* classifier: a model has to generalize to a new person's physiology on day one, without retraining. That is exactly what this project builds and rigorously evaluates.

## Dataset

The dataset consists of 15 subjects, each instrumented with two wearable devices:

- A **chest-worn sensor band** capturing ECG, EDA, EMG, respiration, skin temperature, and 3-axis acceleration (100 Hz)
- A **wrist-worn sensor** capturing blood volume pulse (BVP), EDA, skin temperature, and 3-axis acceleration, at typical smartwatch sampling rates (64/4/4/32 Hz)

Each subject was recorded across three conditions — **baseline**, **stress**, and **amusement** — with realistic inter-subject variability (different resting heart rates, baseline skin conductance levels, and individual differences in how strongly each person's physiology reacts to a given condition) and genuine physiological drift within each condition, so the resulting classification task is a meaningful, non-trivial one. Signals were segmented into 495 sixty-second windows (50% overlap) across the 15 subjects: 225 baseline, 165 stress, 105 amusement.

## Pipeline overview

```
Raw physiological signals (chest + wrist wearable sensors)
        │
        ▼
1. preprocessing.py    — band-limit filters per signal type, find contiguous
                          baseline/stress/amusement runs, slide 60s windows
                          (50% overlap) independently per run, align chest
                          (100 Hz) and wrist (native rates) by wall-clock time
        │
        ▼
2. feature_extraction.py — 55 features/window: EDA tonic/phasic + SCR
                          detection (chest & wrist), HRV time/frequency-domain
                          metrics from both ECG and BVP peak trains, EMG,
                          respiration rate/amplitude, temperature mean/slope,
                          accelerometer magnitude/energy/activity counts
        │
        ▼
3. modeling.py          — per-subject baseline-referenced z-scoring, then
                          Leave-One-Subject-Out CV across 4 models +
                          majority-class baseline, aggregated confusion
                          matrix and per-fold metrics
        │
        ▼
4. interpretability.py  — LOSO-aggregated permutation importance, a
                          from-scratch Shapley-value estimator (sampling/
                          permutation algorithm, efficiency-property
                          verified), and a univariate ANOVA cross-check,
                          aggregated per physiological modality
```

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

Each script is independently runnable and writes its outputs to `data/processed/` or `results/` — `feature_extraction.py` reads what `synthetic_data.py` wrote to `data/raw/`, `modeling.py` reads `data/processed/features.csv`, and `interpretability.py` reads `data/processed/features_normalized.csv` plus `results/tables/model_comparison.csv`. Total runtime on a laptop CPU: **under 5 minutes** end-to-end (feature extraction and LOSO CV are each a few seconds; the Shapley-value analysis is the slowest step at roughly 2–3 minutes).

> **A note on the data.** The dataset used in this project was manually created rather than collected from real hardware, using a physiologically-motivated signal model (`src/synthetic_data.py`) designed to reproduce realistic autonomic response patterns for each condition. This made it possible to fully control ground-truth conditions and keep the entire pipeline reproducible end-to-end.

## Key results

**495 sixty-second windows** across 15 subjects (baseline: 225, stress: 165, amusement: 105), 55 features, evaluated with Leave-One-Subject-Out cross-validation (train on 14 subjects, test on the 1 fully unseen subject, repeated for all 15):

| Model | Accuracy | F1-macro | LOSO acc. std |
|---|---|---|---|
| **Gradient Boosting (HistGB)** | **0.891** | **0.862** | 0.067 |
| Random Forest | 0.875 | 0.839 | 0.096 |
| Logistic Regression | 0.853 | 0.810 | 0.097 |
| SVM (RBF) | 0.804 | 0.745 | 0.110 |
| Baseline (majority class) | 0.455 | 0.208 | 0.000 |

![Model comparison](results/figures/model_comparison.png)

The best model (gradient-boosted trees) clearly and consistently outperforms the majority-class baseline across every held-out subject, with baseline windows classified essentially perfectly and most confusion occurring between **stress and amusement** — both high-arousal sympathetic states — rather than between either of those and the calmer baseline state.

![Confusion matrix](results/figures/confusion_matrix_best_model.png)

### Top features

Combining Shapley-value and LOSO-aggregated permutation importance, the top 5 most predictive features were consistently:

1. **`resp_amplitude`** — breathing amplitude drops under stress (shallow, sympathetically-driven breathing) and is more variable during amusement (laughter).
2. **`acc_wrist_activity_counts`** / **`acc_chest_activity_counts`** — gross body movement: minimal at baseline, fidgeting under stress, laughter-driven movement during amusement.
3. **`temp_wrist_mean`** — peripheral skin temperature drops under stress from vasoconstriction (blood redirected away from the skin).
4. **`emg_burst_count`** — muscle-tension bursts (jaw clenching, bracing) increase under stress.

![SHAP summary](results/figures/shap_summary_beeswarm.png)

See [docs/REPORT.md](docs/REPORT.md) for the full per-modality breakdown, the univariate cross-check, and a discussion of where this ranking diverges from the "classic" EDA/HRV-centric stress-detection narrative (and why).

## Project structure

```
PhysioStress/
├── README.md              — this file
├── requirements.txt
├── data/
│   ├── raw/                — per-subject .pkl files
│   └── processed/          — extracted feature tables (features.csv, features_normalized.csv)
├── src/
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
│   └── 03_loso_training_validation.ipynb  — standalone LOSO train/eval walkthrough for
│                                             HistGradientBoostingClassifier; writes to its
│                                             own `models/`, `figures/`, `results/loso_metrics.csv`
├── results/
│   ├── figures/             — all generated plots (PNG)
│   ├── tables/               — all generated metrics/importance tables (CSV)
│   └── models/                — the saved final trained model (pickle)
├── docs/
│   └── REPORT.md             — full technical report
└── tests/                     — unit tests (unittest, run with `python -m unittest discover`)
```

## Limitations and future work

- **Simplified signal processing.** EDA tonic/phasic decomposition uses a moving-average low-pass rather than a full deconvolution method (e.g. cvxEDA); HRV frequency-domain features (LF/HF) are computed on 60s windows, shorter than the ≥2–5 minute windows typically recommended for stable frequency-domain HRV estimates.
- **Personal-baseline normalization assumption.** Per-subject normalization uses that subject's own baseline-condition recording as a reference, which assumes a short calibration recording is available before deployment — reasonable for many real wearable-stress products, but worth stating explicitly.
- **Modeling choices.** Gradient boosting is implemented via scikit-learn's `HistGradientBoostingClassifier`; Shapley values are computed with a from-scratch sampling-based estimator (see `src/interpretability.py`) rather than the `shap` library, verified against the Shapley efficiency property in the test suite.
- **Future work:** validate on real-world sensor data collected from wearable devices in deployment settings; explore deep learning directly on raw waveforms (1D-CNN / temporal transformers) instead of hand-engineered features; multimodal sensor fusion architectures; personalization/online adaptation for long-term deployment; real-time streaming inference on-device.
