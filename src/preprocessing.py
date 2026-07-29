"""
preprocessing.py
-----------------
Loading, cleaning, segmenting, and windowing of physiological recordings.

Pipeline for one subject:
  1. Load the raw recording (`load_subject`) from `data/raw/{subject_id}.pkl`.
  2. Band-limit each raw signal with an appropriate filter
     (`clean_chest_signals` / `clean_wrist_signals`) to remove drift and
     high-frequency noise before any feature is computed.
  3. Find contiguous same-label runs in the chest label channel
     (`find_label_runs`) and keep only baseline / stress / amusement runs,
     dropping transient/undefined/meditation segments.
  4. Slide fixed-length windows (60s, 50% overlap by default) over each
     run independently (`window_subject`), never letting a window span
     two different labels. Chest and wrist windows are aligned by
     wall-clock time within the run, since the two devices sample at
     different rates.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from utils import (
    CHEST_SIGNALS, DATA_RAW_DIR, FS, LABEL_MAP, WINDOW_OVERLAP, WINDOW_SEC,
    WRIST_SIGNALS, load_pickle, window_indices,
)

KEPT_LABEL_CODES = set(LABEL_MAP.keys())  # {1, 2, 3}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _load_raw_sensor_pkl(subject_id: str, data_dir: str) -> Optional[dict]:
    """
    Load one subject's raw sensor recording from `data/raw/{subject_id}.pkl`
    (dict with 'signal' -> {'chest':..., 'wrist':...}, 'label', 'subject').
    Returns None if the file isn't present.

    Chest signals ('ACC','ECG','EMG','EDA','Temp','Resp') are stored at a
    shared chest sampling rate; wrist signals ('ACC', 'BVP', 'EDA', 'TEMP')
    at their own device-specific rates. ACC channels are (N, 3) arrays.
    `label` is a chest-rate-aligned integer array (0=undefined, 1=baseline,
    2=stress, 3=amusement, 4=meditation, 5/6/7=other transient).
    """
    path = os.path.join(data_dir, f"{subject_id}.pkl")
    if not os.path.exists(path):
        return None
    raw = load_pickle(path)

    chest_raw = raw["signal"]["chest"]
    wrist_raw = raw["signal"]["wrist"]
    label = np.asarray(raw["label"]).astype(np.int64).ravel()

    def _split_acc(acc_arr):
        acc_arr = np.asarray(acc_arr)
        return acc_arr[:, 0], acc_arr[:, 1], acc_arr[:, 2]

    cx, cy, cz = _split_acc(chest_raw["ACC"])
    chest = {
        "ECG": np.asarray(chest_raw["ECG"]).ravel(),
        "EDA": np.asarray(chest_raw["EDA"]).ravel(),
        "EMG": np.asarray(chest_raw["EMG"]).ravel(),
        "RESP": np.asarray(chest_raw["Resp"]).ravel(),
        "TEMP": np.asarray(chest_raw["Temp"]).ravel(),
        "ACC_x": cx, "ACC_y": cy, "ACC_z": cz,
    }
    wx, wy, wz = _split_acc(wrist_raw["ACC"])
    wrist = {
        "BVP": np.asarray(wrist_raw["BVP"]).ravel(),
        "EDA": np.asarray(wrist_raw["EDA"]).ravel(),
        "TEMP": np.asarray(wrist_raw["TEMP"]).ravel(),
        "ACC_x": wx, "ACC_y": wy, "ACC_z": wz,
    }
    return {
        "subject_id": subject_id,
        "chest": chest,
        "chest_label": label,
        "chest_fs": 700,
        "wrist": wrist,
        "wrist_fs": {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32},
        "segment_times": None,
        "is_synthetic": False,
    }


def load_subject(subject_id: str, data_dir: str = DATA_RAW_DIR) -> dict:
    """Load one subject's recording, generated dataset first, falling back
    to a raw sensor pickle if present."""
    synth_path = os.path.join(data_dir, f"{subject_id}_synthetic.pkl")
    if os.path.exists(synth_path):
        return load_pickle(synth_path)
    raw = _load_raw_sensor_pkl(subject_id, data_dir)
    if raw is not None:
        return raw
    raise FileNotFoundError(
        f"No data found for subject '{subject_id}' in {data_dir}. Expected either "
        f"'{subject_id}_synthetic.pkl' (run src/synthetic_data.py first) or "
        f"'{subject_id}.pkl' (a raw sensor pickle in the same format)."
    )


def data_source_label(subject_data: dict) -> str:
    return "generated" if subject_data.get("is_synthetic", True) else "raw sensor file"


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def _butter_filter(sig: np.ndarray, fs: float, low: Optional[float], high: Optional[float],
                    order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    if low is not None and high is not None:
        low_n, high_n = max(low / nyq, 1e-4), min(high / nyq, 0.999)
        if low_n >= high_n:
            return sig
        b, a = butter(order, [low_n, high_n], btype="band")
    elif high is not None:
        b, a = butter(order, min(high / nyq, 0.999), btype="low")
    elif low is not None:
        b, a = butter(order, max(low / nyq, 1e-4), btype="high")
    else:
        return sig
    if len(sig) <= 3 * max(len(a), len(b)):
        return sig  # too short to filtfilt safely
    return filtfilt(b, a, sig)


def clean_chest_signals(chest: Dict[str, np.ndarray], fs: float) -> Dict[str, np.ndarray]:
    """Band-limit each chest channel to its physiologically relevant band."""
    out = dict(chest)
    out["ECG"] = _butter_filter(chest["ECG"], fs, 0.5, 40)
    out["EDA"] = _butter_filter(chest["EDA"], fs, None, 5)
    out["EMG"] = _butter_filter(chest["EMG"], fs, 20, min(45, fs / 2 - 1))
    out["RESP"] = _butter_filter(chest["RESP"], fs, 0.1, 0.8)
    # TEMP and ACC left unfiltered (slow / already physically band-limited).
    return out


def clean_wrist_signals(wrist: Dict[str, np.ndarray], wrist_fs: Dict[str, float]) -> Dict[str, np.ndarray]:
    out = dict(wrist)
    out["BVP"] = _butter_filter(wrist["BVP"], wrist_fs["BVP"], 0.5, min(8, wrist_fs["BVP"] / 2 - 1))
    # Wrist EDA/TEMP already sampled at only 4Hz (Nyquist=2Hz) by the real
    # E4 device's own internal filtering -- no further filtering applied.
    return out


# --------------------------------------------------------------------------
# Label runs & windowing
# --------------------------------------------------------------------------
def find_label_runs(label: np.ndarray) -> List[Tuple[int, int, int]]:
    """Run-length encode a label array -> list of (start_idx, end_idx, code)."""
    if len(label) == 0:
        return []
    runs = []
    start = 0
    cur = label[0]
    for i in range(1, len(label)):
        if label[i] != cur:
            runs.append((start, i, int(cur)))
            start = i
            cur = label[i]
    runs.append((start, len(label), int(cur)))
    return runs


def window_subject(subject_data: dict, window_sec: float = WINDOW_SEC,
                    overlap: float = WINDOW_OVERLAP) -> List[dict]:
    """
    Clean, segment, and window one subject's recording.
    Returns a list of window records:
      {subject_id, label (str), chest: {sig: array}, wrist: {sig: array},
       chest_fs, wrist_fs, t_start, t_end}
    """
    stride_sec = window_sec * (1 - overlap)
    chest_fs = subject_data["chest_fs"]
    wrist_fs = subject_data["wrist_fs"]

    chest_clean = clean_chest_signals(subject_data["chest"], chest_fs)
    wrist_clean = clean_wrist_signals(subject_data["wrist"], wrist_fs)
    label = subject_data["chest_label"]

    runs = find_label_runs(label)
    records = []
    for (start_idx, end_idx, code) in runs:
        if code not in KEPT_LABEL_CODES:
            continue  # drop transient / undefined / meditation / other
        label_name = LABEL_MAP[code]
        run_len = end_idx - start_idx
        run_t0 = start_idx / chest_fs  # absolute time (s) of run start in session

        for (w_lo, w_hi) in window_indices(run_len, chest_fs, window_sec, stride_sec):
            abs_lo, abs_hi = start_idx + w_lo, start_idx + w_hi
            t_start = abs_lo / chest_fs
            t_end = abs_hi / chest_fs

            chest_win = {sig: chest_clean[sig][abs_lo:abs_hi] for sig in CHEST_SIGNALS}

            wrist_win = {}
            for sig in WRIST_SIGNALS:
                base_sig = "ACC" if sig.startswith("ACC") else sig
                fs_w = wrist_fs[base_sig]
                lo_w = int(round(t_start * fs_w))
                hi_w = int(round(t_end * fs_w))
                arr = wrist_clean[sig]
                hi_w = min(hi_w, len(arr))
                wrist_win[sig] = arr[lo_w:hi_w]

            records.append({
                "subject_id": subject_data["subject_id"],
                "label": label_name,
                "chest": chest_win,
                "wrist": wrist_win,
                "chest_fs": chest_fs,
                "wrist_fs": wrist_fs,
                "t_start": t_start,
                "t_end": t_end,
            })
    return records


def build_all_windows(subject_ids: List[str], data_dir: str = DATA_RAW_DIR,
                       verbose: bool = True) -> List[dict]:
    """Load + window every subject, returning a flat list of window records."""
    all_records = []
    for sid in subject_ids:
        sdata = load_subject(sid, data_dir)
        recs = window_subject(sdata)
        all_records.extend(recs)
        if verbose:
            counts = {}
            for r in recs:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            print(f"  {sid} [{data_source_label(sdata)}]: {len(recs)} windows "
                  f"({counts})")
    return all_records


if __name__ == "__main__":
    from utils import SUBJECT_IDS
    recs = build_all_windows(SUBJECT_IDS)
    print(f"\nTotal windows across all subjects: {len(recs)}")
