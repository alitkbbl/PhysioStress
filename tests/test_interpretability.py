"""Unit tests for src/interpretability.py."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from interpretability import (
    shapley_values_sampling, verify_shapley_efficiency, aggregate_by_modality,
    univariate_discriminability,
)


class TestShapleyValues(unittest.TestCase):
    def test_efficiency_property_holds_for_linear_function(self):
        """For a simple linear predict_fn, Shapley values have a closed
        form (coef_i * (x_i - background_i)) and must sum exactly to
        f(x) - f(background)."""
        rng = np.random.default_rng(0)
        n_features = 6
        coefs = rng.normal(0, 1, n_features)

        def predict_fn(Z):
            return Z @ coefs

        background = rng.normal(0, 1, size=(20, n_features))
        X = rng.normal(0, 1, size=(8, n_features))

        shap_vals = shapley_values_sampling(predict_fn, X, background, n_samples=60,
                                             rng=np.random.default_rng(1))
        check = verify_shapley_efficiency(shap_vals, predict_fn, X, background)
        # Linear function -> Shapley values should reconstruct the output
        # essentially exactly (small residual only from finite background sampling).
        self.assertLess(check["mean_abs_error"], 0.05 * (check["mean_abs_target"] + 1e-6) + 0.05)

    def test_shap_matrix_shape(self):
        rng = np.random.default_rng(2)
        n_features = 5
        coefs = rng.normal(0, 1, n_features)
        predict_fn = lambda Z: Z @ coefs
        background = rng.normal(0, 1, size=(10, n_features))
        X = rng.normal(0, 1, size=(4, n_features))
        shap_vals = shapley_values_sampling(predict_fn, X, background, n_samples=20)
        self.assertEqual(shap_vals.shape, (4, n_features))

    def test_zero_contribution_for_irrelevant_feature(self):
        """A feature predict_fn never looks at should get ~0 Shapley value."""
        rng = np.random.default_rng(3)
        n_features = 4

        def predict_fn(Z):
            return Z[:, 0] * 2.0  # only feature 0 matters

        background = rng.normal(0, 1, size=(25, n_features))
        X = rng.normal(0, 1, size=(6, n_features))
        shap_vals = shapley_values_sampling(predict_fn, X, background, n_samples=80,
                                             rng=np.random.default_rng(4))
        # features 1,2,3 should have near-zero attribution
        self.assertTrue(np.all(np.abs(shap_vals[:, 1:]) < 0.3))
        # feature 0 should have non-trivial attribution
        self.assertTrue(np.any(np.abs(shap_vals[:, 0]) > 0.3))


class TestAggregateByModality(unittest.TestCase):
    def test_groups_features_correctly(self):
        importance = pd.Series({
            "eda_chest_mean": 0.5, "eda_chest_std": 0.3,
            "hrv_ecg_mean_hr": 0.8, "acc_wrist_mag_std": 0.2,
        })
        agg = aggregate_by_modality(importance)
        self.assertAlmostEqual(agg["EDA (chest)"], 0.8)
        self.assertAlmostEqual(agg["HRV (chest ECG)"], 0.8)
        self.assertAlmostEqual(agg["Accelerometer (wrist)"], 0.2)


class TestUnivariateDiscriminability(unittest.TestCase):
    def test_detects_clearly_separated_feature(self):
        rng = np.random.default_rng(5)
        n = 60
        df = pd.DataFrame({
            "subject_id": [f"S{i % 5}" for i in range(3 * n)],
            "label": ["baseline"] * n + ["stress"] * n + ["amusement"] * n,
            "separated": np.concatenate([rng.normal(0, 1, n), rng.normal(10, 1, n), rng.normal(5, 1, n)]),
            "noise_only": rng.normal(0, 1, 3 * n),
        })
        result = univariate_discriminability(df, ["separated", "noise_only"])
        sep_f = result.loc[result.feature == "separated", "F_stat"].iloc[0]
        noise_f = result.loc[result.feature == "noise_only", "F_stat"].iloc[0]
        self.assertGreater(sep_f, noise_f)
        self.assertGreater(sep_f, 50)  # should be a very large F-stat given full separation


if __name__ == "__main__":
    unittest.main()
