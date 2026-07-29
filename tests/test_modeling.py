"""Unit tests for src/modeling.py."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modeling import (
    get_feature_columns, normalize_per_subject_baseline, build_models, run_loso_cv,
)


def make_toy_dataframe(n_subjects=4, n_per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        subj_offset = rng.normal(0, 5)  # subject-level bias to be normalized away
        for label, class_mean in [("baseline", 0.0), ("stress", 4.0), ("amusement", 2.0)]:
            for _ in range(n_per_class):
                rows.append({
                    "subject_id": f"S{s}",
                    "label": label,
                    "feat_a": subj_offset + class_mean + rng.normal(0, 0.5),
                    "feat_b": subj_offset * 2 + class_mean * 0.5 + rng.normal(0, 0.5),
                })
    return pd.DataFrame(rows)


class TestFeatureColumns(unittest.TestCase):
    def test_excludes_subject_and_label(self):
        df = make_toy_dataframe()
        cols = get_feature_columns(df)
        self.assertNotIn("subject_id", cols)
        self.assertNotIn("label", cols)
        self.assertIn("feat_a", cols)
        self.assertIn("feat_b", cols)


class TestNormalizePerSubjectBaseline(unittest.TestCase):
    def test_baseline_windows_centered_near_zero_per_subject(self):
        df = make_toy_dataframe()
        cols = get_feature_columns(df)
        norm = normalize_per_subject_baseline(df, cols)
        for sid, g in norm.groupby("subject_id"):
            base_mean = g.loc[g.label == "baseline", "feat_a"].mean()
            self.assertAlmostEqual(base_mean, 0.0, delta=0.35)

    def test_removes_subject_level_offset(self):
        # Before normalization, subject-level offsets should create large
        # variance in feat_a across subjects even within the same class;
        # after normalization that variance should shrink substantially.
        df = make_toy_dataframe(n_subjects=6, n_per_class=8, seed=1)
        cols = get_feature_columns(df)
        norm = normalize_per_subject_baseline(df, cols)

        stress_raw = df.loc[df.label == "stress"].groupby("subject_id")["feat_a"].mean()
        stress_norm = norm.loc[norm.label == "stress"].groupby("subject_id")["feat_a"].mean()
        self.assertLess(stress_norm.std(), stress_raw.std())

    def test_no_nan_or_inf_in_output(self):
        df = make_toy_dataframe()
        cols = get_feature_columns(df)
        norm = normalize_per_subject_baseline(df, cols)
        self.assertFalse(norm[cols].isna().any().any())
        self.assertTrue(np.isfinite(norm[cols].values).all())

    def test_handles_zero_variance_baseline_without_crashing(self):
        df = make_toy_dataframe(n_subjects=2, n_per_class=3)
        cols = get_feature_columns(df)
        # force zero variance in one subject's baseline feature
        mask = (df.subject_id == "S0") & (df.label == "baseline")
        df.loc[mask, "feat_a"] = 1.0
        norm = normalize_per_subject_baseline(df, cols)
        self.assertTrue(np.isfinite(norm[cols].values).all())


class TestBuildModels(unittest.TestCase):
    def test_returns_at_least_three_real_models_plus_baseline(self):
        models = build_models()
        self.assertIn("Baseline (Majority Class)", models)
        non_baseline = [k for k in models if k != "Baseline (Majority Class)"]
        self.assertGreaterEqual(len(non_baseline), 3)

    def test_all_models_are_fittable_pipelines(self):
        df = make_toy_dataframe()
        cols = get_feature_columns(df)
        X = df[cols].values
        y = df["label"].values
        models = build_models()
        for name, pipe in models.items():
            pipe.fit(X, y)
            preds = pipe.predict(X)
            self.assertEqual(len(preds), len(y))


class TestRunLosoCv(unittest.TestCase):
    def test_every_subject_held_out_exactly_once(self):
        df = make_toy_dataframe(n_subjects=5, n_per_class=6, seed=2)
        cols = get_feature_columns(df)
        norm = normalize_per_subject_baseline(df, cols)
        models = {"LR": build_models()["Logistic Regression"]}
        results = run_loso_cv(norm, cols, models, verbose=False)
        fold_df = results["LR"]["fold_results"]
        self.assertEqual(sorted(fold_df["subject_id"].tolist()), sorted(df["subject_id"].unique().tolist()))
        self.assertEqual(fold_df["n_test"].sum(), len(df))

    def test_majority_baseline_never_beats_random_chance_by_much(self):
        df = make_toy_dataframe(n_subjects=5, n_per_class=6, seed=3)
        cols = get_feature_columns(df)
        norm = normalize_per_subject_baseline(df, cols)
        models = {"Dummy": build_models()["Baseline (Majority Class)"]}
        results = run_loso_cv(norm, cols, models, verbose=False)
        # 3 balanced classes -> majority baseline accuracy should be close to 1/3
        self.assertAlmostEqual(results["Dummy"]["overall_accuracy"], 1 / 3, delta=0.05)

    def test_informative_model_beats_majority_baseline(self):
        df = make_toy_dataframe(n_subjects=6, n_per_class=8, seed=4)
        cols = get_feature_columns(df)
        norm = normalize_per_subject_baseline(df, cols)
        models = build_models()
        results = run_loso_cv(norm, cols, models, verbose=False)
        dummy_acc = results["Baseline (Majority Class)"]["overall_accuracy"]
        rf_acc = results["Random Forest"]["overall_accuracy"]
        self.assertGreater(rf_acc, dummy_acc)


if __name__ == "__main__":
    unittest.main()
