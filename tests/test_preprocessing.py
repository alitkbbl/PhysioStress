"""Unit tests for src/preprocessing.py."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import find_label_runs, window_subject, clean_chest_signals
from utils import window_indices, WINDOW_SEC, FS


class TestWindowIndices(unittest.TestCase):
    def test_basic_windowing_no_overlap_edge(self):
        # 100 Hz, 60s windows, 50% overlap -> stride 30s -> stride_samples=3000
        idx = window_indices(n_samples=6000, fs=100, window_sec=60, stride_sec=30)
        self.assertEqual(idx, [(0, 6000)])  # only one full window fits (needs 6000+3000 for 2)

    def test_multiple_windows(self):
        # 480s of data at 100Hz, 60s window/50% overlap -> 15 windows expected
        idx = window_indices(n_samples=48000, fs=100, window_sec=60, stride_sec=30)
        self.assertEqual(len(idx), 15)
        # every window should be exactly 6000 samples long
        for lo, hi in idx:
            self.assertEqual(hi - lo, 6000)

    def test_too_short_returns_empty(self):
        idx = window_indices(n_samples=100, fs=100, window_sec=60, stride_sec=30)
        self.assertEqual(idx, [])

    def test_no_negative_or_out_of_range_indices(self):
        idx = window_indices(n_samples=48000, fs=100, window_sec=60, stride_sec=30)
        for lo, hi in idx:
            self.assertGreaterEqual(lo, 0)
            self.assertLessEqual(hi, 48000)
            self.assertLess(lo, hi)


class TestFindLabelRuns(unittest.TestCase):
    def test_simple_runs(self):
        label = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0])
        runs = find_label_runs(label)
        self.assertEqual(runs, [(0, 2, 0), (2, 5, 1), (5, 7, 2), (7, 9, 0)])

    def test_single_run(self):
        label = np.full(10, 3)
        runs = find_label_runs(label)
        self.assertEqual(runs, [(0, 10, 3)])

    def test_empty(self):
        self.assertEqual(find_label_runs(np.array([])), [])

    def test_runs_cover_full_array_contiguously(self):
        rng = np.random.default_rng(0)
        label = rng.integers(0, 4, size=500)
        runs = find_label_runs(label)
        # runs must be contiguous and cover [0, len(label))
        self.assertEqual(runs[0][0], 0)
        self.assertEqual(runs[-1][1], len(label))
        for (s1, e1, _), (s2, e2, _) in zip(runs, runs[1:]):
            self.assertEqual(e1, s2)
        # reconstruct and compare
        recon = np.zeros_like(label)
        for s, e, code in runs:
            recon[s:e] = code
        np.testing.assert_array_equal(recon, label)


class TestCleanChestSignals(unittest.TestCase):
    def test_output_shapes_and_types_preserved(self):
        fs = 100
        n = fs * 60
        rng = np.random.default_rng(0)
        chest = {
            "ECG": rng.normal(0, 1, n), "EDA": rng.normal(3, 0.5, n), "EMG": rng.normal(0, 0.1, n),
            "RESP": rng.normal(0, 1, n), "TEMP": np.full(n, 36.5), "ACC_x": rng.normal(0, 0.1, n),
            "ACC_y": rng.normal(0, 0.1, n), "ACC_z": np.full(n, 1.0),
        }
        cleaned = clean_chest_signals(chest, fs)
        for k in chest:
            self.assertIn(k, cleaned)
            self.assertEqual(len(cleaned[k]), n)
            self.assertTrue(np.all(np.isfinite(cleaned[k])))

    def test_filtering_reduces_low_frequency_drift_in_ecg(self):
        # A large slow drift (baseline wander) should be substantially
        # attenuated by the 0.5-40Hz ECG bandpass.
        fs = 100
        n = fs * 60
        t = np.arange(n) / fs
        drift = 5.0 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz, well below 0.5Hz cutoff
        chest = {"ECG": drift, "EDA": np.ones(n), "EMG": np.zeros(n), "RESP": np.zeros(n),
                 "TEMP": np.ones(n), "ACC_x": np.zeros(n), "ACC_y": np.zeros(n), "ACC_z": np.ones(n)}
        cleaned = clean_chest_signals(chest, fs)
        self.assertLess(np.std(cleaned["ECG"]), 0.3 * np.std(drift))


class TestWindowSubject(unittest.TestCase):
    def _make_fake_subject(self):
        fs = 100
        rng = np.random.default_rng(1)
        # 90s baseline, 10s transient(label0), 90s stress
        n1, n_gap, n2 = 90 * fs, 10 * fs, 90 * fs
        n_total = n1 + n_gap + n2
        chest = {k: rng.normal(0, 1, n_total) for k in
                 ["ECG", "EDA", "EMG", "RESP", "TEMP", "ACC_x", "ACC_y", "ACC_z"]}
        label = np.concatenate([np.full(n1, 1), np.full(n_gap, 0), np.full(n2, 2)])
        wrist = {
            "BVP": rng.normal(0, 1, int(n_total / fs * 64)),
            "EDA": rng.normal(0, 1, int(n_total / fs * 4)),
            "TEMP": rng.normal(0, 1, int(n_total / fs * 4)),
            "ACC_x": rng.normal(0, 1, int(n_total / fs * 32)),
            "ACC_y": rng.normal(0, 1, int(n_total / fs * 32)),
            "ACC_z": rng.normal(0, 1, int(n_total / fs * 32)),
        }
        return {
            "subject_id": "TEST1", "chest": chest, "chest_label": label, "chest_fs": fs,
            "wrist": wrist, "wrist_fs": {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32},
            "segment_times": None, "is_synthetic": True,
        }

    def test_drops_transient_label(self):
        subj = self._make_fake_subject()
        records = window_subject(subj, window_sec=60, overlap=0.5)
        labels = {r["label"] for r in records}
        self.assertNotIn("undefined", labels)
        self.assertTrue(labels.issubset({"baseline", "stress", "amusement"}))

    def test_expected_window_count(self):
        subj = self._make_fake_subject()
        records = window_subject(subj, window_sec=60, overlap=0.5)
        # each 90s run -> floor((90-60)/30)+1 = 2 windows; two runs -> 4 total
        self.assertEqual(len(records), 4)

    def test_chest_window_length_matches_spec(self):
        subj = self._make_fake_subject()
        records = window_subject(subj, window_sec=60, overlap=0.5)
        for r in records:
            self.assertEqual(len(r["chest"]["ECG"]), 60 * 100)

    def test_wrist_window_length_approximately_matches_duration(self):
        subj = self._make_fake_subject()
        records = window_subject(subj, window_sec=60, overlap=0.5)
        for r in records:
            # 60s at 64Hz = 3840 samples, allow small off-by-one from rounding
            self.assertAlmostEqual(len(r["wrist"]["BVP"]), 60 * 64, delta=2)
            self.assertAlmostEqual(len(r["wrist"]["EDA"]), 60 * 4, delta=2)


if __name__ == "__main__":
    unittest.main()
