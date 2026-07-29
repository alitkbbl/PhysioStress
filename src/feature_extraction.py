"""
feature_extraction.py
----------------------
Per-window feature extraction for each physiological signal modality.
Every function below documents the physiological rationale for the
features it produces (this is later echoed in docs/REPORT.md).

Feature groups (column-name prefixes map to a "modality" in
utils.feature_to_modality, used for the interpretability aggregation):
  eda_chest_ / eda_wrist_   Electrodermal activity: tonic (SCL, slow
                            sympathetic arousal baseline) + phasic
                            (SCRs, fast bursts tied to discrete sympathetic
                            "events" -- more frequent/larger under stress).
  hrv_ecg_  / hrv_bvp_      Heart-rate variability, from chest ECG R-peaks
                            and wrist BVP pulse-peaks respectively. Lower
                            variability (SDNN/RMSSD) and higher LF/HF is a
                            classic marker of sympathetic dominance (stress).
  emg_                      Trapezius EMG: muscle tension increases with
                            stress-related bracing/jaw clenching.
  resp_                     Respiration: faster, shallower breathing under
                            stress (sympathetic activation).
  temp_chest_ / temp_wrist_ Peripheral vasoconstriction under stress
                            diverts blood from skin -> skin temperature
                            drifts down; it recovers/rises at rest.
  acc_chest_ / acc_wrist_   Gross body movement -- captures fidgeting
                            (stress) or laughter-related motion (amusement)
                            versus stillness (baseline).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

EPS = 1e-8


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def _safe(val, default=np.nan):
    if val is None:
        return default
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return default
    return val


def _linear_slope(x: np.ndarray, fs: float) -> float:
    if len(x) < 2:
        return np.nan
    t = np.arange(len(x)) / fs
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, x, rcond=None)[0]
    return float(slope)


def _moving_average(x: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    win = max(1, int(round(window_sec * fs)))
    if win >= len(x):
        return np.full_like(x, np.mean(x)) if len(x) > 0 else x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


# --------------------------------------------------------------------------
# EDA: tonic/phasic decomposition + SCR detection
# --------------------------------------------------------------------------
def eda_features(eda: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    """
    Simplified tonic/phasic decomposition: tonic (skin conductance level,
    SCL) is approximated with a 10s moving average low-pass; phasic
    (skin conductance responses, SCRs) is the residual. This is a
    lightweight stand-in for more sophisticated deconvolution methods
    (e.g. cvxEDA, neurokit2's `eda_process`) -- documented as a limitation.
    SCR peaks are then detected on the phasic residual with a minimum
    prominence and minimum spacing (refractory period).
    """
    feats = {}
    if len(eda) < 4:
        keys = ["mean", "std", "slope", "tonic_mean", "tonic_slope",
                 "phasic_std", "scr_count", "scr_amp_mean", "scr_amp_max", "phasic_auc"]
        return {f"{prefix}{k}": np.nan for k in keys}

    tonic = _moving_average(eda, fs, window_sec=min(10.0, len(eda) / fs))
    phasic = eda - tonic

    peaks, props = find_peaks(phasic, prominence=max(0.02, 0.1 * (phasic.std() + EPS)),
                               distance=max(1, int(1.0 * fs)))
    amps = props["prominences"] if len(peaks) else np.array([])

    feats[f"{prefix}mean"] = float(np.mean(eda))
    feats[f"{prefix}std"] = float(np.std(eda))
    feats[f"{prefix}slope"] = _linear_slope(eda, fs)
    feats[f"{prefix}tonic_mean"] = float(np.mean(tonic))
    feats[f"{prefix}tonic_slope"] = _linear_slope(tonic, fs)
    feats[f"{prefix}phasic_std"] = float(np.std(phasic))
    feats[f"{prefix}scr_count"] = float(len(peaks))
    feats[f"{prefix}scr_amp_mean"] = float(np.mean(amps)) if len(amps) else 0.0
    feats[f"{prefix}scr_amp_max"] = float(np.max(amps)) if len(amps) else 0.0
    feats[f"{prefix}phasic_auc"] = float(np.sum(np.clip(phasic, 0, None)) / fs)
    return {k: _safe(v) for k, v in feats.items()}


# --------------------------------------------------------------------------
# HRV: peak detection (ECG R-peaks or BVP pulse-peaks) + time/freq metrics
# --------------------------------------------------------------------------
def _detect_peaks(sig: np.ndarray, fs: float, min_bpm: float = 40, max_bpm: float = 200) -> np.ndarray:
    if len(sig) < fs:  # need at least ~1s of data
        return np.array([], dtype=int)
    sig = sig - np.median(sig)
    rng = np.ptp(sig)
    if rng <= 0:
        return np.array([], dtype=int)
    height = 0.35 * np.max(sig)
    min_distance = max(1, int(round(60.0 / max_bpm * fs)))
    peaks, _ = find_peaks(sig, height=height, distance=min_distance)
    return peaks


def _lf_hf_ratio(peak_times_sec: np.ndarray) -> float:
    if len(peak_times_sec) < 5:
        return np.nan
    rr = np.diff(peak_times_sec)  # seconds
    t_rr = peak_times_sec[1:]
    fs_interp = 4.0
    if t_rr[-1] - t_rr[0] < 20:  # need a reasonable span to resolve LF band
        return np.nan
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_interp)
    if len(t_uniform) < 16:
        return np.nan
    interp = np.interp(t_uniform, t_rr, rr * 1000.0)
    interp = interp - interp.mean()
    window = np.hanning(len(interp))
    spec = np.fft.rfft(interp * window)
    psd = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(interp), d=1.0 / fs_interp)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    lf_power = psd[lf_mask].sum()
    hf_power = psd[hf_mask].sum()
    if hf_power <= 0:
        return np.nan
    return float(lf_power / hf_power)


def hrv_features(sig: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    keys = ["mean_hr", "sdnn", "rmssd", "pnn50", "lf_hf", "n_beats"]
    peaks = _detect_peaks(sig, fs)
    if len(peaks) < 4:
        return {f"{prefix}{k}": np.nan for k in keys}

    peak_times = peaks / fs
    rr = np.diff(peak_times)  # seconds
    rr_ms = rr * 1000.0
    hr = 60.0 / rr

    mean_hr = float(np.mean(hr))
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan
    diffs = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else np.nan
    pnn50 = float(100.0 * np.sum(np.abs(diffs) > 50) / len(diffs)) if len(diffs) > 0 else np.nan
    lf_hf = _lf_hf_ratio(peak_times)

    feats = {
        f"{prefix}mean_hr": mean_hr, f"{prefix}sdnn": sdnn, f"{prefix}rmssd": rmssd,
        f"{prefix}pnn50": pnn50, f"{prefix}lf_hf": lf_hf, f"{prefix}n_beats": float(len(peaks)),
    }
    return {k: _safe(v) for k, v in feats.items()}


# --------------------------------------------------------------------------
# EMG
# --------------------------------------------------------------------------
def emg_features(emg: np.ndarray, fs: float, prefix: str = "emg_") -> Dict[str, float]:
    if len(emg) < 4:
        keys = ["rms", "std", "mav", "zcr", "burst_count"]
        return {f"{prefix}{k}": np.nan for k in keys}
    rms = float(np.sqrt(np.mean(emg ** 2)))
    std = float(np.std(emg))
    mav = float(np.mean(np.abs(emg)))
    zcr = float(np.mean(np.diff(np.sign(emg)) != 0))
    env = np.abs(emg)
    thresh = np.mean(env) + 2 * np.std(env)
    above = env > thresh
    burst_count = float(np.sum(np.diff(above.astype(int)) == 1))
    return {f"{prefix}rms": rms, f"{prefix}std": std, f"{prefix}mav": mav,
            f"{prefix}zcr": zcr, f"{prefix}burst_count": burst_count}


# --------------------------------------------------------------------------
# Respiration
# --------------------------------------------------------------------------
def resp_features(resp: np.ndarray, fs: float, prefix: str = "resp_") -> Dict[str, float]:
    keys = ["rate_bpm", "amplitude", "std", "rate_var"]
    if len(resp) < fs * 5:
        return {f"{prefix}{k}": np.nan for k in keys}
    sig = resp - np.mean(resp)
    peaks, props = find_peaks(sig, prominence=max(0.05, 0.2 * (sig.std() + EPS)),
                               distance=max(1, int(1.2 * fs)))
    duration_min = len(resp) / fs / 60.0
    rate_bpm = float(len(peaks) / duration_min) if duration_min > 0 else np.nan
    amplitude = float(props["prominences"].mean()) if len(peaks) else float(np.std(sig))
    rate_var = float(np.std(np.diff(peaks) / fs)) if len(peaks) > 2 else np.nan
    return {f"{prefix}rate_bpm": _safe(rate_bpm), f"{prefix}amplitude": _safe(amplitude),
            f"{prefix}std": float(np.std(resp)), f"{prefix}rate_var": _safe(rate_var)}


# --------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------
def temp_features(temp: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    if len(temp) < 2:
        return {f"{prefix}mean": np.nan, f"{prefix}slope": np.nan, f"{prefix}std": np.nan}
    return {
        f"{prefix}mean": float(np.mean(temp)),
        f"{prefix}slope": _linear_slope(temp, fs),
        f"{prefix}std": float(np.std(temp)),
    }


# --------------------------------------------------------------------------
# Accelerometer
# --------------------------------------------------------------------------
def acc_features(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray, fs: float,
                  prefix: str) -> Dict[str, float]:
    keys = ["mag_mean", "mag_std", "energy", "activity_counts"]
    if len(acc_x) < 4:
        return {f"{prefix}{k}": np.nan for k in keys}
    mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    energy = float(np.mean(mag ** 2))
    centered = mag - _moving_average(mag, fs, window_sec=min(2.0, len(mag) / fs))
    thresh = 1.5 * (np.std(centered) + EPS)
    activity_counts = float(np.sum(np.abs(centered) > thresh))
    return {f"{prefix}mag_mean": mag_mean, f"{prefix}mag_std": mag_std,
            f"{prefix}energy": energy, f"{prefix}activity_counts": activity_counts}


# --------------------------------------------------------------------------
# Master extraction for one window record (see preprocessing.window_subject)
# --------------------------------------------------------------------------
def extract_features_for_window(win: dict) -> Dict[str, float]:
    chest_fs = win["chest_fs"]
    wrist_fs = win["wrist_fs"]
    feats = {"subject_id": win["subject_id"], "label": win["label"]}

    feats.update(eda_features(win["chest"]["EDA"], chest_fs, "eda_chest_"))
    feats.update(eda_features(win["wrist"]["EDA"], wrist_fs["EDA"], "eda_wrist_"))
    feats.update(hrv_features(win["chest"]["ECG"], chest_fs, "hrv_ecg_"))
    feats.update(hrv_features(win["wrist"]["BVP"], wrist_fs["BVP"], "hrv_bvp_"))
    feats.update(emg_features(win["chest"]["EMG"], chest_fs, "emg_"))
    feats.update(resp_features(win["chest"]["RESP"], chest_fs, "resp_"))
    feats.update(temp_features(win["chest"]["TEMP"], chest_fs, "temp_chest_"))
    feats.update(temp_features(win["wrist"]["TEMP"], wrist_fs["TEMP"], "temp_wrist_"))
    feats.update(acc_features(win["chest"]["ACC_x"], win["chest"]["ACC_y"], win["chest"]["ACC_z"],
                               chest_fs, "acc_chest_"))
    feats.update(acc_features(win["wrist"]["ACC_x"], win["wrist"]["ACC_y"], win["wrist"]["ACC_z"],
                               wrist_fs["ACC"], "acc_wrist_"))
    return feats


def build_feature_dataframe(window_records: List[dict], verbose: bool = True) -> pd.DataFrame:
    rows = []
    for i, win in enumerate(window_records):
        rows.append(extract_features_for_window(win))
        if verbose and (i + 1) % 100 == 0:
            print(f"  extracted features for {i + 1}/{len(window_records)} windows")
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    from utils import SUBJECT_IDS, DATA_PROCESSED_DIR
    from preprocessing import build_all_windows
    import os

    print("Building windows...")
    records = build_all_windows(SUBJECT_IDS)
    print(f"\nExtracting features for {len(records)} windows...")
    df = build_feature_dataframe(records)
    out_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved feature table: {out_path}  shape={df.shape}")
    print(f"NaN count per column (top 10):\n{df.isna().sum().sort_values(ascending=False).head(10)}")
