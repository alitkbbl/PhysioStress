"""
modeling.py
------------
Model definitions and the Leave-One-Subject-Out (LOSO) cross-validation
protocol used to evaluate them.

WHY LOSO AND NOT RANDOM K-FOLD?
Random k-fold CV would place windows from the *same subject* (and often
adjacent, highly-correlated windows thanks to 50% overlap) in both the
train and test splits. Because individual physiology is highly
idiosyncratic (resting HR, EDA baseline level, etc. vary a lot between
people), a model can partly "memorize" a subject's signature from the
training windows and then trivially recognize the held-out windows from
the *same* subject -- inflating accuracy in a way that will not
generalize to a new person wearing the device. Leave-One-Subject-Out CV
is standard practice for exactly this reason in subject-independent
wearable-sensor classification: every fold trains on 14 subjects and
tests on the 1 completely unseen subject, which is the realistic
deployment scenario (a new user, whose data the model has never touched).

PER-SUBJECT NORMALIZATION
Feature values (e.g. resting HR, baseline EDA level) differ substantially
between people for reasons unrelated to stress. Every subject's features
are therefore z-scored using the mean/std of *that same subject's own
baseline-condition windows* before any model sees them. This is a
personal calibration step (assumes a short baseline recording is
available per user) and uses only that subject's own data -- it does not
leak information from other subjects or from the label being predicted,
so it is safe to apply identically whether a subject ends up in the
train or the test fold.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import LABEL_NAMES, RANDOM_SEED

NON_FEATURE_COLS = {"subject_id", "label"}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# --------------------------------------------------------------------------
# Per-subject baseline normalization
# --------------------------------------------------------------------------
def normalize_per_subject_baseline(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Z-score every subject's features using that subject's own baseline stats."""
    out = df.copy()
    for sid, group in df.groupby("subject_id"):
        base_mask = group["label"] == "baseline"
        mu = group.loc[base_mask, feature_cols].mean()
        sd = group.loc[base_mask, feature_cols].std().replace(0, np.nan).fillna(1.0)
        sd = sd.where(sd > 1e-8, 1e-8)
        idx = group.index
        out.loc[idx, feature_cols] = (group[feature_cols] - mu) / sd
    # guard against any residual inf/nan from degenerate subjects
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
def build_models(seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    At least 3 real candidate models plus a majority-class dummy baseline.
    Gradient boosting uses scikit-learn's HistGradientBoostingClassifier
    (a histogram-based gradient-boosted tree ensemble, comparable in
    spirit to XGBoost/LightGBM).
    """
    models = {
        "Baseline (Majority Class)": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                        random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                            class_weight="balanced", random_state=seed,
                                            n_jobs=-1)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced",
                        probability=True, random_state=seed)),
        ]),
        "Gradient Boosting (HistGB)": Pipeline([
            ("clf", HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                    learning_rate=0.08,
                                                    class_weight="balanced",
                                                    random_state=seed)),
        ]),
    }
    return models


# --------------------------------------------------------------------------
# LOSO cross-validation
# --------------------------------------------------------------------------
def run_loso_cv(df: pd.DataFrame, feature_cols: List[str], models: Dict[str, Pipeline],
                 label_names: List[str] = LABEL_NAMES, verbose: bool = True) -> dict:
    """
    Run Leave-One-Subject-Out CV for every model in `models`.
    Returns a dict: {model_name: {"fold_results": DataFrame(subject, acc, f1_macro),
                                    "confusion_matrix": np.ndarray,
                                    "overall_accuracy": float, "overall_f1_macro": float,
                                    "y_true": [...], "y_pred": [...]}}
    """
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["subject_id"].values
    logo = LeaveOneGroupOut()

    results = {}
    for name, pipeline in models.items():
        fold_rows = []
        all_true, all_pred = [], []
        for train_idx, test_idx in logo.split(X, y, groups):
            held_out_subject = groups[test_idx][0]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = _clone_pipeline(pipeline)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro", labels=label_names, zero_division=0)
            fold_rows.append({"subject_id": held_out_subject, "n_test": len(test_idx),
                               "accuracy": acc, "f1_macro": f1m})
            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())

        fold_df = pd.DataFrame(fold_rows)
        cm = confusion_matrix(all_true, all_pred, labels=label_names)
        overall_acc = accuracy_score(all_true, all_pred)
        overall_f1 = f1_score(all_true, all_pred, average="macro", labels=label_names, zero_division=0)

        results[name] = {
            "fold_results": fold_df,
            "confusion_matrix": cm,
            "overall_accuracy": overall_acc,
            "overall_f1_macro": overall_f1,
            "y_true": all_true,
            "y_pred": all_pred,
        }
        if verbose:
            print(f"{name:28s}  acc={overall_acc:.3f}  f1_macro={overall_f1:.3f}  "
                  f"(per-subject acc std={fold_df['accuracy'].std():.3f})")
    return results


def _clone_pipeline(pipeline: Pipeline) -> Pipeline:
    from sklearn.base import clone
    return clone(pipeline)


def results_summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "accuracy": r["overall_accuracy"],
            "f1_macro": r["overall_f1_macro"],
            "loso_acc_std": r["fold_results"]["accuracy"].std(),
            "loso_f1_std": r["fold_results"]["f1_macro"].std(),
        })
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import os
    from utils import DATA_PROCESSED_DIR, TABLES_DIR, save_pickle, MODELS_DIR, set_all_seeds
    from plots import (plot_class_distribution, plot_model_comparison,
                        plot_confusion_matrix, plot_per_subject_accuracy)

    set_all_seeds()
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "features.csv"))
    feature_cols = get_feature_columns(df)
    print(f"Loaded {df.shape[0]} windows, {len(feature_cols)} features, "
          f"{df['subject_id'].nunique()} subjects")
    print("Class distribution:", df["label"].value_counts().to_dict())

    df_norm = normalize_per_subject_baseline(df, feature_cols)
    df_norm.to_csv(os.path.join(DATA_PROCESSED_DIR, "features_normalized.csv"), index=False)

    models = build_models()
    print("\nRunning Leave-One-Subject-Out cross-validation...\n")
    results = run_loso_cv(df_norm, feature_cols, models)

    summary = results_summary_table(results)
    print("\n=== Model comparison (sorted by F1-macro) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(TABLES_DIR, "model_comparison.csv"), index=False)

    # Save per-fold results and confusion matrices for every model.
    all_fold_rows = []
    for name, r in results.items():
        fr = r["fold_results"].copy()
        fr["model"] = name
        all_fold_rows.append(fr)
        cm_df = pd.DataFrame(r["confusion_matrix"], index=LABEL_NAMES, columns=LABEL_NAMES)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        cm_df.to_csv(os.path.join(TABLES_DIR, f"confusion_matrix_{safe_name}.csv"))
    pd.concat(all_fold_rows, ignore_index=True).to_csv(
        os.path.join(TABLES_DIR, "loso_fold_results.csv"), index=False)

    # Figures
    plot_class_distribution(df)
    plot_model_comparison(summary)
    best_model_name = summary.iloc[0]["model"]
    best = results[best_model_name]
    plot_confusion_matrix(best["confusion_matrix"], LABEL_NAMES,
                           title=f"Confusion matrix — {best_model_name} (LOSO, aggregated)",
                           name="confusion_matrix_best_model.png")
    plot_per_subject_accuracy(best["fold_results"], best_model_name)

    print(f"\nBest model: {best_model_name}  "
          f"(accuracy={best['overall_accuracy']:.3f}, F1-macro={best['overall_f1_macro']:.3f})")
    print(f"Saved tables to {TABLES_DIR} and figures to results/figures/")
