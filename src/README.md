# `src/` — Pipeline

---

## Flow
```text
Raw physiological signals (chest + wrist wearable sensors)
│
▼
1. preprocessing.py       — band-limit filters per signal type, find contiguous
baseline/stress/amusement runs, slide 60s windows
(50% overlap) independently per run, align chest
(100 Hz) and wrist (native rates) by wall-clock time
│
▼
2. feature_extraction.py  — 55 features/window: EDA tonic/phasic + SCR
detection (chest & wrist), HRV time/frequency-domain
metrics from both ECG and BVP peak trains, EMG,
respiration rate/amplitude, temperature mean/slope,
accelerometer magnitude/energy/activity counts
│
▼
3. modeling.py            — per-subject baseline-referenced z-scoring, then
Leave-One-Subject-Out CV across 4 models +
majority-class baseline, aggregated confusion
matrix and per-fold metrics
│
▼
4. interpretability.py    — LOSO-aggregated permutation importance, a
from-scratch Shapley-value estimator (sampling/
permutation algorithm, efficiency-property
verified), and a univariate ANOVA cross-check,
aggregated per physiological modality
```

> Each stage is an **independently runnable** script (`python3 src/<stage>.py`) that reads the previous stage's output from `data/` or `results/` and writes its own — see the main [README](../README.md#installation--how-to-run) for the exact run order.

---

## Modules

| Module                  | Responsibility |
|:-------------------------|:----------------|
| `utils.py`               | Shared config: paths, sampling rates, label maps, windowing helpers, the feature→modality mapping used throughout interpretability. |
| `synthetic_data.py`      | Generates the dataset (see `data/README.md`) — beat-to-beat ECG/PPG, tonic+phasic EDA, bursty EMG, respiration, temperature drift, accelerometer traces, with per-subject reactivity and within-condition physiological drift. |
| `preprocessing.py`       | Loads a subject, applies signal-appropriate Butterworth filters, finds contiguous condition runs, and slides 60s/50%-overlap windows aligned across the two devices. |
| `feature_extraction.py`  | Computes all 55 per-window features (EDA, HRV from both ECG and BVP, EMG, respiration, temperature, accelerometer) — see the module docstring for the physiological rationale behind each group. |
| `modeling.py`            | Per-subject baseline normalization, the model zoo (Logistic Regression, Random Forest, SVM, HistGradientBoostingClassifier, majority-class baseline), and the Leave-One-Subject-Out CV loop. |
| `interpretability.py`    | LOSO-aggregated permutation importance, the from-scratch Shapley-value sampling estimator (with an efficiency-property correctness check), and the univariate ANOVA cross-check. |
| `plots.py`               | Shared matplotlib/seaborn figure styling used by `modeling.py` and `interpretability.py`. |

---

## Why Leave-One-Subject-Out *(not random k-fold)*

Random k-fold CV would let a model partly "memorize" a subject's individual signature (resting HR, baseline EDA level, etc.) from training windows and recognize held-out windows from the *same* subject — inflating accuracy in a way that won't generalize to a new person.

**LOSO** (train on 14 subjects, test on the 1 unseen subject, repeated for all 15) directly measures what matters: accuracy on someone the model has never seen.

---

## Full Methodology

This file covers **what** each module does; **why** each design choice was made — feature rationale, normalization justification, model selection, evaluation protocol, interpretability methodology — is written up in full in [`docs/README.md`](../docs/README.md).
`
