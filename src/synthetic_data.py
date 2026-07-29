"""
synthetic_data.py
------------------
Physiological signal generator for the PhysioStress dataset: 15 subjects,
each wearing a chest sensor band (ECG, EDA, EMG, respiration, temperature,
3-axis acceleration) and a wrist sensor (BVP, EDA, temperature, 3-axis
acceleration), recorded across baseline / stress / amusement conditions.

Signals are built from simple, documented generative models (beat-to-beat
ECG/PPG pulse trains, tonic+phasic EDA, bursty EMG, sinusoidal respiration,
drifting temperature, and noisy accelerometer traces), parameterized so
that stress/baseline/amusement segments differ in the *direction* real
autonomic physiology is known to move (e.g. higher heart rate and lower
heart-rate-variability under stress, elevated EDA tonic level under
stress, faster/shallower respiration under stress).

Design choices (documented for transparency):
- 15 subjects each get randomized individual traits (resting HR, baseline
  EDA level, per-signal stress *reactivity*, etc.) so that genuine
  inter-subject variability exists -- this is what makes Leave-One-
  Subject-Out cross-validation and per-subject normalization meaningful
  rather than vacuous.
- Chest signals are generated at 100 Hz, wrist signals at typical
  wrist-wearable rates (BVP 64 Hz, EDA/TEMP 4 Hz, ACC 32 Hz).
- Segment durations (8 / 6 / 4 minutes for baseline / stress / amusement)
  are chosen to give several 60s windows per subject per condition while
  keeping total pipeline runtime fast.
- Short 15s "transient" gaps (label 0) are inserted between segments, so
  `preprocessing.py`'s label-cleaning logic (dropping undefined/transient
  periods) has genuine transitions to exercise.
- Within-segment physiological drift (mean-reverting trends) and
  per-subject reactivity multipliers ensure the resulting classification
  task is realistically difficult rather than trivially separable -- see
  docs/REPORT.md Section 2 for the full rationale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from utils import (
    CHEST_SIGNALS, DATA_RAW_DIR, FS, LABEL_TO_INT, N_SUBJECTS,
    SUBJECT_IDS, SYNTHETIC_DURATION_SEC, WRIST_SIGNALS, save_pickle,
)

TRANSIENT_GAP_SEC = 15
CONDITIONS = ["baseline", "stress", "amusement"]


# --------------------------------------------------------------------------
# Low-level signal primitives
# --------------------------------------------------------------------------
def _add_gaussian_pulse(sig: np.ndarray, t: np.ndarray, fs: int, center: float,
                         amp: float, width: float) -> None:
    """Add a local Gaussian pulse to `sig` in-place (only touches nearby samples)."""
    idx_center = int(round(center * fs))
    half_span = max(1, int(round(5 * width * fs)))
    lo, hi = max(0, idx_center - half_span), min(len(sig), idx_center + half_span)
    if lo >= hi:
        return
    tt = t[lo:hi]
    sig[lo:hi] += amp * np.exp(-0.5 * ((tt - center) / width) ** 2)


def _add_ppg_pulse(sig: np.ndarray, t: np.ndarray, fs: int, center: float,
                    amp: float, w_rise: float, w_decay: float) -> None:
    """Asymmetric fast-rise/slow-decay pulse, roughly PPG-shaped."""
    idx_center = int(round(center * fs))
    half_span = max(1, int(round(5 * w_decay * fs)))
    lo, hi = max(0, idx_center - half_span), min(len(sig), idx_center + half_span)
    if lo >= hi:
        return
    tt = t[lo:hi] - center
    pulse = np.where(tt < 0,
                      amp * np.exp(-0.5 * (tt / w_rise) ** 2),
                      amp * np.exp(-0.5 * (tt / w_decay) ** 2))
    sig[lo:hi] += pulse


def _ou_trend(duration_sec: float, target: float, sigma: float, tau: float,
              rng: np.random.Generator, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean-reverting Ornstein-Uhlenbeck trend, evaluated at coarse resolution
    `dt` (seconds) and meant to be linearly interpolated afterwards. Models
    realistic minute-to-minute physiological drift *within* a single
    condition segment (real autonomic state is not perfectly constant for
    several minutes) -- this is what makes different 60s windows drawn from
    the same condition genuinely differ from one another, rather than all
    windows in a segment being near-identical draws around one fixed value
    (which would make the classification task unrealistically easy).
    `target` is the long-run mean the process reverts to; `sigma` is the
    (approximate) standard deviation of the stationary distribution around
    that mean; `tau` is the reversion time constant in seconds.
    """
    n = max(2, int(round(duration_sec / dt)) + 1)
    x = np.empty(n)
    x[0] = target + rng.normal(0, sigma)
    diffusion = sigma * np.sqrt(2 * dt / tau)
    for i in range(1, n):
        x[i] = x[i - 1] + (target - x[i - 1]) * (dt / tau) + rng.normal(0, diffusion)
    t_grid = np.arange(n) * dt
    return t_grid, x


def _trend_fn(t_grid: np.ndarray, values: np.ndarray):
    return lambda t: np.interp(t, t_grid, values)


def _generate_beat_times(duration_sec: float, hr_fn, rr_std_ms_fn,
                          resp_freq: float, rng: np.random.Generator) -> np.ndarray:
    """Beat-to-beat times (s), with time-varying mean HR / HRV trends,
    respiratory sinus arrhythmia, and per-beat noise."""
    beats = []
    t = 0.0
    while t < duration_sec:
        local_hr = max(35.0, hr_fn(t))
        mean_rr = 60.0 / local_hr
        rsa = 0.05 * mean_rr * np.sin(2 * np.pi * resp_freq * t)
        noise = rng.normal(0, max(rr_std_ms_fn(t), 3.0) / 1000.0)
        rr = max(0.28, mean_rr + rsa + noise)  # clip to <~215 bpm
        beats.append(t)
        t += rr
    return np.array(beats)


def _simulate_ecg(duration_sec: float, fs: int, hr_fn, rr_std_ms_fn,
                   resp_freq: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    ecg = np.zeros(n)
    beats = _generate_beat_times(duration_sec, hr_fn, rr_std_ms_fn, resp_freq, rng)
    for bt in beats:
        _add_gaussian_pulse(ecg, t, fs, bt - 0.16, 0.15, 0.022)   # P wave
        _add_gaussian_pulse(ecg, t, fs, bt, 1.0, 0.010)           # R spike
        _add_gaussian_pulse(ecg, t, fs, bt + 0.20, 0.25, 0.045)   # T wave
    ecg += rng.normal(0, 0.03, n)
    return ecg, beats


def _simulate_bvp(duration_sec: float, fs: int, hr_fn, rr_std_ms_fn,
                   resp_freq: float, rng: np.random.Generator) -> np.ndarray:
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    bvp = np.zeros(n)
    # Independent beat train (pulse-transit-time + sensor differences vs ECG)
    rr_std_wrist_fn = lambda tt: rr_std_ms_fn(tt) * 1.15
    beats = _generate_beat_times(duration_sec, hr_fn, rr_std_wrist_fn, resp_freq, rng)
    for bt in beats:
        jitter = rng.normal(0, 0.01)
        _add_ppg_pulse(bvp, t, fs, bt + jitter, 1.0, 0.06, 0.16)
    bvp += rng.normal(0, 0.05, n)
    return bvp


def _simulate_eda(duration_sec: float, fs: int, tonic_fn, tonic_drift_std: float,
                   scr_rate_per_min: float, scr_amp_range: Tuple[float, float],
                   rng: np.random.Generator) -> np.ndarray:
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    tonic_center = tonic_fn(t)
    tonic = tonic_center + np.cumsum(rng.normal(0, tonic_drift_std, n))
    tonic = np.clip(tonic, 0.3, None)
    phasic = np.zeros(n)
    n_scr = rng.poisson(max(scr_rate_per_min * duration_sec / 60.0, 0))
    onsets = np.sort(rng.uniform(0, duration_sec, n_scr))
    for onset in onsets:
        amp = rng.uniform(*scr_amp_range)
        rise, decay = rng.uniform(0.5, 1.2), rng.uniform(1.5, 4.0)
        idx0 = int(onset * fs)
        span = int(8 * decay * fs)
        lo, hi = idx0, min(n, idx0 + span)
        if lo >= hi:
            continue
        tt = np.arange(lo, hi) / fs - onset
        pulse = amp * (1 - np.exp(-tt / rise)) * np.exp(-tt / decay)
        phasic[lo:hi] += pulse
    eda = tonic + phasic + rng.normal(0, 0.015, n)
    return np.clip(eda, 0, None)


def _simulate_emg(duration_sec: float, fs: int, base_std_fn,
                   burst_rate_per_min: float, burst_amp_range: Tuple[float, float],
                   rng: np.random.Generator) -> np.ndarray:
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    local_std = np.clip(base_std_fn(t), 0.005, None)
    emg = rng.normal(0, 1, n) * local_std
    n_burst = rng.poisson(max(burst_rate_per_min * duration_sec / 60.0, 0))
    onsets = rng.uniform(0, duration_sec, n_burst)
    for onset in onsets:
        dur = rng.uniform(0.4, 1.8)
        idx0, idx1 = int(onset * fs), min(n, int(onset * fs) + int(dur * fs))
        if idx0 >= idx1:
            continue
        amp = rng.uniform(*burst_amp_range)
        emg[idx0:idx1] += rng.normal(0, amp, idx1 - idx0)
    return emg


def _simulate_resp(duration_sec: float, fs: int, rate_fn,
                    amplitude: float, rng: np.random.Generator) -> np.ndarray:
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    inst_freq = np.clip(rate_fn(t), 4, 40) / 60.0
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    phase_noise = np.cumsum(rng.normal(0, 0.015, n)) / fs
    resp = amplitude * np.sin(phase + phase_noise)
    resp += rng.normal(0, 0.04 * amplitude, n)
    return resp


def _simulate_temp(duration_sec: float, fs: int, base_temp: float,
                    slope_per_sec: float, noise_std: float,
                    rng: np.random.Generator) -> np.ndarray:
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    drift = np.cumsum(rng.normal(0, noise_std, n)) * 0.03
    temp = base_temp + slope_per_sec * t + drift
    return temp


def _simulate_acc(duration_sec: float, fs: int, base_vec: np.ndarray, noise_std_fn,
                   burst_rate_per_min: float, burst_amp: float,
                   rng: np.random.Generator) -> np.ndarray:
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    local_noise_std = np.clip(noise_std_fn(t), 0.003, None)
    acc = np.tile(base_vec, (n, 1)).astype(float)
    acc += rng.normal(0, 1, (n, 3)) * local_noise_std[:, None]
    n_burst = rng.poisson(max(burst_rate_per_min * duration_sec / 60.0, 0))
    onsets = rng.uniform(0, duration_sec, n_burst)
    for onset in onsets:
        dur = rng.uniform(1.0, 3.5)
        idx0, idx1 = int(onset * fs), min(n, int(onset * fs) + int(dur * fs))
        if idx0 >= idx1:
            continue
        freq = rng.uniform(1.0, 3.0)
        tt = np.arange(idx1 - idx0) / fs
        osc = burst_amp * np.sin(2 * np.pi * freq * tt)[:, None] * rng.uniform(0.5, 1.5, size=(1, 3))
        acc[idx0:idx1] += osc
    return acc


# --------------------------------------------------------------------------
# Subject-level traits & condition parameter tables
# --------------------------------------------------------------------------
@dataclass
class SubjectTraits:
    hr_base: float
    hrv_base_ms: float
    eda_tonic_base_chest: float
    eda_tonic_base_wrist: float
    eda_reactivity: float          # multiplier on SCR amplitude/rate
    emg_base_std: float
    resp_rate_base: float
    temp_base_chest: float
    temp_base_wrist: float
    acc_posture_chest: np.ndarray
    acc_posture_wrist: np.ndarray
    # Individual differences in autonomic *reactivity*: how strongly this
    # person's signals actually shift under stress/amusement, independent
    # of their resting baseline level. Real people vary substantially in
    # how "visible" their stress response is to a wearable -- some barely
    # react physiologically under the same subjective stress, others react
    # strongly. This is what per-subject baseline normalization *cannot*
    # remove (normalization only re-centers/rescales using the subject's
    # own baseline distribution; it does not know how far that subject
    # will move under stress), so it is the main realistic source of
    # cross-subject (LOSO) generalization error in this simulation.
    hr_reactivity: float
    hrv_reactivity: float
    emg_reactivity: float
    resp_reactivity: float
    temp_reactivity: float
    acc_reactivity: float


def _generate_subject_traits(rng: np.random.Generator) -> SubjectTraits:
    return SubjectTraits(
        hr_base=rng.normal(72, 7),
        hrv_base_ms=rng.uniform(55, 95),
        eda_tonic_base_chest=rng.uniform(2.0, 7.0),
        eda_tonic_base_wrist=rng.uniform(1.0, 4.5),
        eda_reactivity=rng.uniform(0.5, 1.6),
        emg_base_std=rng.uniform(0.015, 0.045),
        resp_rate_base=rng.normal(14.5, 1.8),
        temp_base_chest=rng.normal(36.6, 0.25),
        temp_base_wrist=rng.normal(33.8, 0.5),
        acc_posture_chest=np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.03, 3),
        acc_posture_wrist=np.array([0.2, -0.1, 0.95]) + rng.normal(0, 0.05, 3),
        hr_reactivity=rng.uniform(0.15, 2.1),
        hrv_reactivity=rng.uniform(0.15, 2.1),
        emg_reactivity=rng.uniform(0.15, 2.1),
        resp_reactivity=rng.uniform(0.15, 2.1),
        temp_reactivity=rng.uniform(0.1, 2.2),
        acc_reactivity=rng.uniform(0.15, 2.1),
    )



# Condition modulation factors relative to each subject's own baseline trait.
# These encode the *direction* of known autonomic stress responses.
_CONDITION_MODS = {
    "baseline": dict(hr_delta=0.0, hrv_mult=1.0, eda_tonic_delta_mult=0.0,
                     scr_rate=2.0, scr_amp=(0.05, 0.25), emg_burst_rate=1.0,
                     emg_burst_amp=(0.03, 0.08), resp_rate_delta=0.0, resp_amp=1.0,
                     temp_slope=0.0002, acc_burst_rate=1.0, acc_burst_amp=0.02,
                     acc_noise=0.015),
    "stress": dict(hr_delta=7.0, hrv_mult=0.78, eda_tonic_delta_mult=0.6,
                   scr_rate=6.5, scr_amp=(0.15, 0.55), emg_burst_rate=2.5,
                   emg_burst_amp=(0.08, 0.20), resp_rate_delta=2.8, resp_amp=0.8,
                   temp_slope=-0.00018, acc_burst_rate=1.3, acc_burst_amp=0.018,
                   acc_noise=0.03),
    "amusement": dict(hr_delta=3.2, hrv_mult=0.96, eda_tonic_delta_mult=0.32,
                      scr_rate=5.0, scr_amp=(0.1, 0.4), emg_burst_rate=1.1,
                      emg_burst_amp=(0.05, 0.12), resp_rate_delta=1.0, resp_amp=0.97,
                      temp_slope=0.00009, acc_burst_rate=2.2, acc_burst_amp=0.030,
                      acc_noise=0.025),
}


def _generate_condition_segment(traits: SubjectTraits, condition: str, duration_sec: float,
                                 rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    mod = _CONDITION_MODS[condition]

    # Long-run per-segment targets (subject trait + condition delta scaled
    # by that subject's own reactivity + a small one-off draw so repeated
    # segments of the same condition for the same subject aren't
    # bit-identical).
    hr_target = max(45.0, traits.hr_base + mod["hr_delta"] * traits.hr_reactivity + rng.normal(0, 2.5))
    hrv_shrink = 1.0 - (1.0 - mod["hrv_mult"]) * traits.hrv_reactivity
    hrv_target = max(8.0, traits.hrv_base_ms * np.clip(hrv_shrink, 0.15, 1.3) * rng.uniform(0.9, 1.1))
    resp_target = max(8.0, traits.resp_rate_base + mod["resp_rate_delta"] * traits.resp_reactivity
                       + rng.normal(0, 1.0))
    emg_target = max(0.005, traits.emg_base_std * (1 + 0.15 * rng.standard_normal()))
    eda_c_target = (traits.eda_tonic_base_chest
                     + mod["eda_tonic_delta_mult"] * 2.5 * traits.eda_reactivity)
    eda_w_target = (traits.eda_tonic_base_wrist
                     + mod["eda_tonic_delta_mult"] * 1.6 * traits.eda_reactivity)
    is_arousal_condition = mod["acc_noise"] > 0.015
    acc_noise_target = mod["acc_noise"] * ((0.6 + 0.6 * traits.acc_reactivity) if is_arousal_condition else 1.0)

    # Mean-reverting trends *within* the segment: real autonomic state
    # drifts minute-to-minute even within one condition, so different 60s
    # windows from the same segment are not near-identical (see
    # `_ou_trend` docstring). Sigma is chosen as a healthy fraction of each
    # target so within-condition variability is comparable in scale to
    # between-condition differences -- this is what keeps the classification
    # task realistically hard rather than trivially perfect.
    hr_fn = _trend_fn(*_ou_trend(duration_sec, hr_target, sigma=11.0, tau=55, rng=rng))
    hrv_fn = _trend_fn(*_ou_trend(duration_sec, hrv_target, sigma=0.35 * hrv_target, tau=70, rng=rng))
    resp_fn = _trend_fn(*_ou_trend(duration_sec, resp_target, sigma=6.0, tau=26, rng=rng))
    emg_fn = _trend_fn(*_ou_trend(duration_sec, emg_target, sigma=0.72 * emg_target, tau=22, rng=rng))
    eda_c_fn = _trend_fn(*_ou_trend(duration_sec, eda_c_target, sigma=2.8, tau=50, rng=rng))
    eda_w_fn = _trend_fn(*_ou_trend(duration_sec, eda_w_target, sigma=1.85, tau=50, rng=rng))
    acc_noise_fn = _trend_fn(*_ou_trend(duration_sec, acc_noise_target,
                                         sigma=0.68 * acc_noise_target, tau=18, rng=rng))
    acc_noise_wrist_fn = _trend_fn(*_ou_trend(duration_sec, acc_noise_target * 1.2,
                                               sigma=0.68 * acc_noise_target, tau=18, rng=rng))

    chest_fs = FS["chest"]
    ecg, _ = _simulate_ecg(duration_sec, chest_fs, hr_fn, hrv_fn, resp_target / 60.0, rng)
    eda_chest = _simulate_eda(
        duration_sec, chest_fs, tonic_fn=eda_c_fn, tonic_drift_std=0.0008,
        scr_rate_per_min=mod["scr_rate"] * traits.eda_reactivity,
        scr_amp_range=tuple(a * traits.eda_reactivity for a in mod["scr_amp"]),
        rng=rng,
    )
    emg = _simulate_emg(duration_sec, chest_fs, emg_fn,
                        mod["emg_burst_rate"] * traits.emg_reactivity, mod["emg_burst_amp"], rng)
    resp = _simulate_resp(duration_sec, chest_fs, resp_fn, mod["resp_amp"], rng)
    temp_chest = _simulate_temp(duration_sec, chest_fs, traits.temp_base_chest,
                                 mod["temp_slope"] * traits.temp_reactivity, 0.05, rng)
    acc_chest = _simulate_acc(duration_sec, chest_fs, traits.acc_posture_chest,
                               acc_noise_fn, mod["acc_burst_rate"] * traits.acc_reactivity,
                               mod["acc_burst_amp"], rng)

    bvp = _simulate_bvp(duration_sec, FS["wrist_bvp"], hr_fn, hrv_fn, resp_target / 60.0, rng)
    eda_wrist = _simulate_eda(
        duration_sec, FS["wrist_eda"], tonic_fn=eda_w_fn, tonic_drift_std=0.001,
        scr_rate_per_min=mod["scr_rate"] * traits.eda_reactivity * 0.85,
        scr_amp_range=tuple(a * traits.eda_reactivity * 0.8 for a in mod["scr_amp"]),
        rng=rng,
    )
    temp_wrist = _simulate_temp(duration_sec, FS["wrist_temp"], traits.temp_base_wrist,
                                 mod["temp_slope"] * 1.3 * traits.temp_reactivity, 0.07, rng)
    acc_wrist = _simulate_acc(duration_sec, FS["wrist_acc"], traits.acc_posture_wrist,
                               acc_noise_wrist_fn, mod["acc_burst_rate"] * traits.acc_reactivity,
                               mod["acc_burst_amp"] * 1.3, rng)

    chest = {"ECG": ecg, "EDA": eda_chest, "EMG": emg, "RESP": resp, "TEMP": temp_chest,
              "ACC_x": acc_chest[:, 0], "ACC_y": acc_chest[:, 1], "ACC_z": acc_chest[:, 2]}
    wrist = {"BVP": bvp, "EDA": eda_wrist, "TEMP": temp_wrist,
              "ACC_x": acc_wrist[:, 0], "ACC_y": acc_wrist[:, 1], "ACC_z": acc_wrist[:, 2]}
    return {"chest": chest, "wrist": wrist}


def _generate_transient(duration_sec: float, traits: SubjectTraits,
                         rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    """Short undefined/transient gap between conditions (label 0)."""
    # Re-use baseline-like generation but shorter, tagged separately so it
    # gets label 0 and is dropped during preprocessing.
    return _generate_condition_segment(traits, "baseline", duration_sec, rng)


def generate_subject(subject_id: str, seed: int) -> dict:
    """Generate one full synthetic subject session (all conditions concatenated)."""
    rng = np.random.default_rng(seed)
    traits = _generate_subject_traits(rng)

    chest_fs = FS["chest"]
    chest_signals = {k: [] for k in CHEST_SIGNALS}
    wrist_signals = {k: [] for k in WRIST_SIGNALS}
    chest_label = []
    segment_times = {}

    t_cursor = 0.0
    order = ["baseline", "stress", "amusement"]
    for i, cond in enumerate(order):
        dur = SYNTHETIC_DURATION_SEC[cond]
        seg = _generate_condition_segment(traits, cond, dur, rng)
        for k in CHEST_SIGNALS:
            chest_signals[k].append(seg["chest"][k])
        for k in WRIST_SIGNALS:
            wrist_signals[k].append(seg["wrist"][k])
        n_chest = len(seg["chest"]["ECG"])
        chest_label.append(np.full(n_chest, LABEL_TO_INT_RAW[cond], dtype=np.int64))
        segment_times[cond] = (t_cursor, t_cursor + dur)
        t_cursor += dur

        if i < len(order) - 1:
            trans = _generate_transient(TRANSIENT_GAP_SEC, traits, rng)
            for k in CHEST_SIGNALS:
                chest_signals[k].append(trans["chest"][k])
            for k in WRIST_SIGNALS:
                wrist_signals[k].append(trans["wrist"][k])
            n_trans = len(trans["chest"]["ECG"])
            chest_label.append(np.full(n_trans, 0, dtype=np.int64))
            t_cursor += TRANSIENT_GAP_SEC

    chest_signals = {k: np.concatenate(v) for k, v in chest_signals.items()}
    wrist_signals = {k: np.concatenate(v) for k, v in wrist_signals.items()}
    chest_label = np.concatenate(chest_label)

    return {
        "subject_id": subject_id,
        "chest": chest_signals,
        "chest_label": chest_label,
        "chest_fs": chest_fs,
        "wrist": wrist_signals,
        "wrist_fs": {"BVP": FS["wrist_bvp"], "EDA": FS["wrist_eda"],
                     "TEMP": FS["wrist_temp"], "ACC": FS["wrist_acc"]},
        "segment_times": segment_times,
        "is_synthetic": True,
    }


# condition label codes: baseline=1, stress=2, amusement=3
LABEL_TO_INT_RAW = {"baseline": 1, "stress": 2, "amusement": 3}


def generate_all_subjects(out_dir: str = DATA_RAW_DIR, n_subjects: int = N_SUBJECTS,
                           base_seed: int = 42, verbose: bool = True) -> list:
    """Generate and save synthetic data for all subjects. Returns list of subject_ids."""
    os.makedirs(out_dir, exist_ok=True)
    subject_ids = SUBJECT_IDS[:n_subjects]
    for i, sid in enumerate(subject_ids):
        data = generate_subject(sid, seed=base_seed + i)
        path = os.path.join(out_dir, f"{sid}_synthetic.pkl")
        save_pickle(data, path)
        if verbose:
            n_chest = len(data["chest_label"])
            print(f"  generated {sid}: {n_chest / data['chest_fs']:.0f}s chest data -> {path}")
    return subject_ids


if __name__ == "__main__":
    print("Generating the PhysioStress dataset...")
    ids = generate_all_subjects()
    print(f"Done. Generated {len(ids)} subjects in {DATA_RAW_DIR}")
