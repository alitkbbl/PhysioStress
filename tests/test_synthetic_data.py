"""Unit tests for src/synthetic_data.py."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic_data import generate_subject, CONDITIONS
from utils import FS, SYNTHETIC_DURATION_SEC


class TestGenerateSubject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = generate_subject("S_TEST", seed=123)

    def test_all_expected_keys_present(self):
        for key in ["subject_id", "chest", "chest_label", "chest_fs", "wrist", "wrist_fs",
                     "segment_times", "is_synthetic"]:
            self.assertIn(key, self.subject)
        self.assertTrue(self.subject["is_synthetic"])

    def test_chest_signals_all_same_length_as_label(self):
        n_label = len(self.subject["chest_label"])
        for name, arr in self.subject["chest"].items():
            self.assertEqual(len(arr), n_label, msg=f"{name} length mismatch")

    def test_label_contains_only_expected_codes(self):
        codes = set(np.unique(self.subject["chest_label"]).tolist())
        self.assertTrue(codes.issubset({0, 1, 2, 3}))

    def test_all_three_target_conditions_present(self):
        codes = set(np.unique(self.subject["chest_label"]).tolist())
        self.assertTrue({1, 2, 3}.issubset(codes))

    def test_total_duration_matches_config(self):
        expected_total = sum(SYNTHETIC_DURATION_SEC.values()) + 2 * 15  # 2 transient gaps
        actual_total = len(self.subject["chest_label"]) / self.subject["chest_fs"]
        self.assertAlmostEqual(actual_total, expected_total, delta=0.5)

    def test_no_nan_or_inf_in_any_signal(self):
        for arr in self.subject["chest"].values():
            self.assertTrue(np.all(np.isfinite(arr)))
        for arr in self.subject["wrist"].values():
            self.assertTrue(np.all(np.isfinite(arr)))

    def test_reproducible_with_same_seed(self):
        s1 = generate_subject("S_A", seed=99)
        s2 = generate_subject("S_A", seed=99)
        np.testing.assert_array_equal(s1["chest"]["ECG"], s2["chest"]["ECG"])
        np.testing.assert_array_equal(s1["chest_label"], s2["chest_label"])

    def test_different_seeds_give_different_signals(self):
        s1 = generate_subject("S_A", seed=1)
        s2 = generate_subject("S_B", seed=2)
        self.assertFalse(np.array_equal(s1["chest"]["ECG"], s2["chest"]["ECG"]))


if __name__ == "__main__":
    unittest.main()
