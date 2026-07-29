"""
utils.py
--------
Shared configuration constants and helper functions used across the
PhysioStress pipeline (preprocessing, feature extraction, modeling,
interpretability).

Keeping these in one place ensures every stage of the pipeline agrees on
sampling rates, window sizes, label encodings, and the mapping from
individual features to physiological "modalities" (used later for the
per-modality interpretability aggregation).
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset-level configuration
# --------------------------------------------------------------------------
# Condition label convention:
# 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
# 4 = meditation, 5/6/7 = other transient recovery segments.
# This project only keeps the three target classes.
LABEL_MAP = {1: "baseline", 2: "stress", 3: "amusement"}
LABEL_NAMES = ["baseline", "stress", "amusement"]
LABEL_TO_INT = {name: i for i, name in enumerate(LABEL_NAMES)}

N_SUBJECTS = 15
SUBJECT_IDS = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]

# Windowing (applied independently within each contiguous labeled segment,
# never across a condition boundary, so a window is always pure-label).
WINDOW_SEC = 60
WINDOW_OVERLAP = 0.5
WINDOW_STRIDE_SEC = WINDOW_SEC * (1 - WINDOW_OVERLAP)

# Sampling rates. Wrist rates match typical smartwatch/Empatica-class device
# rates. Chest signals are generated at 100 Hz to keep runtime/memory
# tractable; this is documented in README/REPORT as a design choice and
# does not change the modeling protocol.
FS = {
    "chest": 100,       # ECG, EDA, EMG, RESP, TEMP, ACC (x,y,z)
    "wrist_bvp": 64,    # typical wrist-PPG sensor rate
    "wrist_eda": 4,     # typical wrist-EDA sensor rate
    "wrist_temp": 4,    # typical wrist-temperature sensor rate
    "wrist_acc": 32,    # typical wrist-accelerometer sensor rate
}

CHEST_SIGNALS = ["ECG", "EDA", "EMG", "RESP", "TEMP", "ACC_x", "ACC_y", "ACC_z"]
WRIST_SIGNALS = ["BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z"]

# Per-condition segment durations (seconds), chosen to give enough length
# for several 60s windows per subject per condition while keeping total
# runtime fast.
SYNTHETIC_DURATION_SEC = {"baseline": 480, "stress": 360, "amusement": 240}

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Feature -> physiological modality mapping
# --------------------------------------------------------------------------
# Populated dynamically by feature_extraction.py (FEATURE_MODALITY_MAP is
# built from the feature name prefixes used there), but we centralize the
# prefix -> modality label mapping here so every module agrees.
MODALITY_PREFIXES = {
    "eda_chest_": "EDA (chest)",
    "eda_wrist_": "EDA (wrist)",
    "hrv_ecg_": "HRV (chest ECG)",
    "hrv_bvp_": "HRV (wrist BVP)",
    "emg_": "EMG (chest)",
    "resp_": "Respiration (chest)",
    "temp_chest_": "Temperature (chest)",
    "temp_wrist_": "Temperature (wrist)",
    "acc_chest_": "Accelerometer (chest)",
    "acc_wrist_": "Accelerometer (wrist)",
}


def feature_to_modality(feature_name: str) -> str:
    """Map a feature column name to a human-readable physiological modality."""
    for prefix, modality in MODALITY_PREFIXES.items():
        if feature_name.startswith(prefix):
            return modality
    return "Other"


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def window_indices(n_samples: int, fs: int, window_sec: float = WINDOW_SEC,
                    stride_sec: float = WINDOW_STRIDE_SEC) -> List[tuple]:
    """
    Return list of (start_idx, end_idx) sample index pairs for fixed-length
    sliding windows over a contiguous single-label segment of length
    n_samples at sampling rate fs. Windows that would run past the end of
    the segment are dropped (no zero-padding), a common convention for
    wearable-sensor preprocessing.
    """
    win_len = int(round(window_sec * fs))
    stride = int(round(stride_sec * fs))
    if win_len <= 0 or n_samples < win_len:
        return []
    starts = range(0, n_samples - win_len + 1, stride)
    return [(s, s + win_len) for s in starts]


@dataclass
class PipelineConfig:
    """Convenience bundle passed around instead of many loose globals."""
    label_names: List[str] = field(default_factory=lambda: list(LABEL_NAMES))
    window_sec: float = WINDOW_SEC
    window_overlap: float = WINDOW_OVERLAP
    fs: Dict[str, int] = field(default_factory=lambda: dict(FS))
    seed: int = RANDOM_SEED
