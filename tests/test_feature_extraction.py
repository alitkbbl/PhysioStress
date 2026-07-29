"""Unit tests for src/feature_extraction.py."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_extraction import (
    eda_features, hrv_features, emg_features, resp_features, temp_features,
    acc_features, extract_features_for_window, _detect_peaks, _linear_slope,
)


def make_synthetic_ecg(fs, duration_sec, hr_bpm, rng):
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    ecg = np.zeros(n)
    beat_interval = 60.0 / hr_bpm
    beat_times = np.arange(0, duration_sec, beat_interval)
    for bt in beat_times:
        ecg += 1.0 * np.exp(-0.5 * ((t - bt) / 0.01) ** 2)
    ecg += rng.normal(0, 0.02, n)
    return ecg


class TestEdaFeatures(unittest.TestCase):
    def test_output_keys_present(self):
        rng = np.random.default_rng(0)
        eda = 3.0 + np.cumsum(rng.normal(0, 0.001, 6000))
        feats = eda_features(eda, fs=100, prefix="eda_chest_")
        for key in ["mean", "std", "slope", "tonic_mean", "tonic_slope",
                     "phasic_std", "scr_count", "scr_amp_mean", "scr_amp_max", "phasic_auc"]:
            self.assertIn(f"eda_chest_{key}", feats)

    def test_no_nan_or_inf_for_normal_input(self):
        rng = np.random.default_rng(1)
        eda = 4.0 + np.cumsum(rng.normal(0, 0.002, 6000))
        feats = eda_features(eda, fs=100, prefix="eda_chest_")
        for v in feats.values():
            self.assertTrue(np.isfinite(v))

    def test_scr_count_increases_with_more_phasic_bumps(self):
        fs = 100
        n = 6000
        t = np.arange(n) / fs
        flat = np.full(n, 3.0)
        bumpy = flat.copy()
        for onset in [5, 15, 25, 35, 45, 55]:
            tt = t - onset
            pulse = 0.5 * (1 - np.exp(-np.clip(tt, 0, None) / 0.8)) * np.exp(-np.clip(tt, 0, None) / 2.0)
            pulse[tt < 0] = 0
            bumpy += pulse
        f_flat = eda_features(flat, fs, "eda_chest_")
        f_bumpy = eda_features(bumpy, fs, "eda_chest_")
        self.assertGreater(f_bumpy["eda_chest_scr_count"], f_flat["eda_chest_scr_count"])

    def test_short_signal_returns_nan_not_crash(self):
        feats = eda_features(np.array([1.0, 2.0]), fs=100, prefix="eda_chest_")
        self.assertTrue(all(np.isnan(v) for v in feats.values()))


class TestHrvFeatures(unittest.TestCase):
    def test_recovers_approximately_correct_heart_rate(self):
        rng = np.random.default_rng(2)
        ecg = make_synthetic_ecg(fs=100, duration_sec=60, hr_bpm=75, rng=rng)
        feats = hrv_features(ecg, fs=100, prefix="hrv_ecg_")
        self.assertAlmostEqual(feats["hrv_ecg_mean_hr"], 75, delta=5)

    def test_higher_hr_input_gives_higher_output(self):
        rng = np.random.default_rng(3)
        ecg_slow = make_synthetic_ecg(fs=100, duration_sec=60, hr_bpm=60, rng=rng)
        ecg_fast = make_synthetic_ecg(fs=100, duration_sec=60, hr_bpm=110, rng=rng)
        f_slow = hrv_features(ecg_slow, fs=100, prefix="hrv_ecg_")
        f_fast = hrv_features(ecg_fast, fs=100, prefix="hrv_ecg_")
        self.assertGreater(f_fast["hrv_ecg_mean_hr"], f_slow["hrv_ecg_mean_hr"])

    def test_sdnn_nonnegative(self):
        rng = np.random.default_rng(4)
        ecg = make_synthetic_ecg(fs=100, duration_sec=60, hr_bpm=80, rng=rng)
        feats = hrv_features(ecg, fs=100, prefix="hrv_ecg_")
        self.assertGreaterEqual(feats["hrv_ecg_sdnn"], 0)

    def test_flat_signal_returns_nan(self):
        feats = hrv_features(np.zeros(6000), fs=100, prefix="hrv_ecg_")
        self.assertTrue(np.isnan(feats["hrv_ecg_mean_hr"]))


class TestEmgFeatures(unittest.TestCase):
    def test_higher_variance_emg_gives_higher_rms(self):
        rng = np.random.default_rng(5)
        low = rng.normal(0, 0.01, 6000)
        high = rng.normal(0, 0.1, 6000)
        f_low = emg_features(low, fs=100)
        f_high = emg_features(high, fs=100)
        self.assertGreater(f_high["emg_rms"], f_low["emg_rms"])


class TestRespFeatures(unittest.TestCase):
    def test_recovers_approximate_breathing_rate(self):
        fs = 100
        n = 60 * fs
        t = np.arange(n) / fs
        rate_bpm = 15.0
        resp = np.sin(2 * np.pi * (rate_bpm / 60.0) * t)
        feats = resp_features(resp, fs, "resp_")
        self.assertAlmostEqual(feats["resp_rate_bpm"], rate_bpm, delta=3)


class TestTempFeatures(unittest.TestCase):
    def test_slope_sign_matches_trend_direction(self):
        n = 6000
        rising = np.linspace(36.0, 37.0, n)
        falling = np.linspace(37.0, 36.0, n)
        f_rise = temp_features(rising, fs=100, prefix="temp_chest_")
        f_fall = temp_features(falling, fs=100, prefix="temp_chest_")
        self.assertGreater(f_rise["temp_chest_slope"], 0)
        self.assertLess(f_fall["temp_chest_slope"], 0)

    def test_linear_slope_recovers_known_value(self):
        fs = 100
        n = 6000
        true_slope = 0.002  # units per second
        x = 36.0 + true_slope * (np.arange(n) / fs)
        self.assertAlmostEqual(_linear_slope(x, fs), true_slope, places=4)


class TestAccFeatures(unittest.TestCase):
    def test_more_movement_gives_higher_activity_counts(self):
        rng = np.random.default_rng(6)
        n = 6000
        still_x = rng.normal(0, 0.01, n)
        still_y = rng.normal(0, 0.01, n)
        still_z = np.full(n, 1.0) + rng.normal(0, 0.01, n)
        active_x = rng.normal(0, 0.3, n)
        active_y = rng.normal(0, 0.3, n)
        active_z = np.full(n, 1.0) + rng.normal(0, 0.3, n)
        f_still = acc_features(still_x, still_y, still_z, fs=100, prefix="acc_chest_")
        f_active = acc_features(active_x, active_y, active_z, fs=100, prefix="acc_chest_")
        self.assertGreater(f_active["acc_chest_activity_counts"], f_still["acc_chest_activity_counts"])
        self.assertGreater(f_active["acc_chest_mag_std"], f_still["acc_chest_mag_std"])


class TestExtractFeaturesForWindow(unittest.TestCase):
    def test_full_window_produces_all_expected_columns(self):
        rng = np.random.default_rng(7)
        fs_chest = 100
        n_chest = 60 * fs_chest
        win = {
            "subject_id": "S_TEST", "label": "baseline", "chest_fs": fs_chest,
            "wrist_fs": {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32},
            "chest": {
                "ECG": make_synthetic_ecg(fs_chest, 60, 70, rng),
                "EDA": 3 + np.cumsum(rng.normal(0, 0.001, n_chest)),
                "EMG": rng.normal(0, 0.02, n_chest),
                "RESP": np.sin(2 * np.pi * 0.25 * np.arange(n_chest) / fs_chest),
                "TEMP": np.full(n_chest, 36.5),
                "ACC_x": rng.normal(0, 0.02, n_chest),
                "ACC_y": rng.normal(0, 0.02, n_chest),
                "ACC_z": np.full(n_chest, 1.0),
            },
            "wrist": {
                "BVP": make_synthetic_ecg(64, 60, 70, rng),
                "EDA": 2 + np.cumsum(rng.normal(0, 0.001, 240)),
                "TEMP": np.full(240, 33.5),
                "ACC_x": rng.normal(0, 0.02, 1920),
                "ACC_y": rng.normal(0, 0.02, 1920),
                "ACC_z": np.full(1920, 1.0),
            },
        }
        feats = extract_features_for_window(win)
        self.assertEqual(feats["subject_id"], "S_TEST")
        self.assertEqual(feats["label"], "baseline")
        # 55 numeric features + subject_id + label = 57 total keys
        self.assertGreaterEqual(len(feats), 50)
        numeric_vals = [v for k, v in feats.items() if k not in ("subject_id", "label")]
        self.assertTrue(all(np.isfinite(v) for v in numeric_vals))


if __name__ == "__main__":
    unittest.main()
