"""
interpretability.py
--------------------
Feature-importance / explainability analysis for the final deployed model.

IMPORTANT NOTE ON SHAP:
Rather than depend on the third-party `shap` package, this module
implements Shapley value estimation *from first principles*, using the
classic sampling/permutation algorithm of Strumbelj & Kononenko (2010,
"An Efficient Explanation of Individual Classifications using Game
Theory") -- the same game-theoretic definition of feature attribution
that SHAP is built on (SHAP's "KernelSHAP" is a weighted-least-squares
reformulation of the same Shapley values; the permutation/sampling
estimator used here converges to the same quantity).
For each instance and each feature, many random feature orderings are
sampled; a feature's contribution in a given ordering is the change in
model output when it is "revealed" (its real value is used) versus
"hidden" (replaced by a background reference value), averaged over
orderings. This is verified below via the Shapley *efficiency property*
(contributions for an instance should sum to
model_output(instance) - model_output(mean background)).

Two importance methods are produced (per the project requirement):
  (a) model-native importance  -- HistGradientBoostingClassifier does not
      expose `feature_importances_`, so for tree ensembles we use
      permutation importance (scikit-learn, model-agnostic, measures the
      drop in held-out macro-F1 when a feature is shuffled) as the
      "native" model-based method, and for Random Forest we additionally
      report its built-in impurity-based `feature_importances_`.
  (b) Shapley values (custom implementation, see above).
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from utils import RANDOM_SEED, feature_to_modality, set_all_seeds


# --------------------------------------------------------------------------
# Custom Shapley value estimator (sampling / permutation method)
# --------------------------------------------------------------------------
def shapley_values_sampling(predict_fn, X: np.ndarray, background: np.ndarray,
                             n_samples: int = 25, rng: np.random.Generator = None) -> np.ndarray:
    """
    Estimate Shapley values for every row of X, for every feature, w.r.t.
    `predict_fn` (should accept a 2D array and return one scalar score per
    row, e.g. predicted probability of the class of interest).

    Algorithm (Strumbelj & Kononenko, 2010): for each instance x and each
    of `n_samples` random feature permutations, walk through the
    permutation, and for each feature compute the marginal effect of
    switching it from a background reference value to its real value in
    x, given the features already "revealed" earlier in the permutation
    are real and the rest are still background. Averaging this marginal
    contribution over many random permutations converges to the exact
    Shapley value.

    `background`: (n_background, n_features) array of reference samples
    used as "the feature is absent" stand-ins (here: a random sample of
    the training distribution).

    Implementation note: naively calling `predict_fn` once per revealed
    feature (as the textbook algorithm reads) means
    n_instances * n_samples * n_features separate model calls, which is
    extremely slow with per-call Python/sklearn overhead. Instead, every
    intermediate "partially revealed" state across all instances and
    permutations is built up front and scored in a *single* batched
    `predict_fn` call, then contributions are computed from the batch of
    predictions. This is mathematically identical to the textbook version,
    just far faster.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n_instances, n_features = X.shape
    n_bg = background.shape[0]
    steps_per_instance = n_samples * (n_features + 1)

    all_states = np.empty((n_instances * steps_per_instance, n_features), dtype=float)
    perms_all = np.empty((n_instances, n_samples, n_features), dtype=int)

    row = 0
    for i in range(n_instances):
        x = X[i]
        for s in range(n_samples):
            perm = rng.permutation(n_features)
            perms_all[i, s] = perm
            z = background[rng.integers(0, n_bg)].copy()
            all_states[row] = z
            row += 1
            for feat in perm:
                z[feat] = x[feat]
                all_states[row] = z
                row += 1

    preds = predict_fn(all_states)  # single batched model call

    shap_vals = np.zeros((n_instances, n_features))
    row = 0
    for i in range(n_instances):
        for s in range(n_samples):
            perm = perms_all[i, s]
            f_prev = preds[row]
            row += 1
            for feat in perm:
                f_curr = preds[row]
                row += 1
                shap_vals[i, feat] += (f_curr - f_prev)
                f_prev = f_curr
        shap_vals[i] /= n_samples
    return shap_vals


def verify_shapley_efficiency(shap_vals: np.ndarray, predict_fn, X: np.ndarray,
                               background: np.ndarray) -> Dict[str, float]:
    """
    Sanity-check the Shapley efficiency property: sum_j phi_j(x) should
    equal predict_fn(x) - mean(predict_fn(background)) for each instance.
    Returns the mean absolute deviation from this identity (should be
    small relative to the typical prediction range).
    """
    baseline = predict_fn(background).mean()
    preds = predict_fn(X)
    lhs = shap_vals.sum(axis=1)
    rhs = preds - baseline
    abs_err = np.abs(lhs - rhs)
    return {
        "mean_abs_error": float(abs_err.mean()),
        "max_abs_error": float(abs_err.max()),
        "mean_abs_target": float(np.abs(rhs).mean()),
    }


# --------------------------------------------------------------------------
# Native (model-based) importance
# --------------------------------------------------------------------------
def native_feature_importance(model, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str], seed: int = RANDOM_SEED) -> pd.Series:
    """
    Returns a pd.Series of native importances indexed by feature name.
    Uses the model's own `feature_importances_` if the final estimator
    exposes one (e.g. RandomForest); otherwise falls back to permutation
    importance computed on the *same data the model was fit on*. Note:
    when used this way (fit and evaluated on the same data), permutation
    importance can look artificially sparse/overfit for flexible models --
    see `loso_permutation_importance` below for the more rigorous,
    held-out version used as the primary "native" method in this project.
    """
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        return pd.Series(vals, index=feature_names).sort_values(ascending=False)

    result = permutation_importance(model, X, y, n_repeats=20, random_state=seed,
                                     scoring="f1_macro", n_jobs=-1)
    return pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)


def loso_permutation_importance(df: pd.DataFrame, feature_cols: List[str], model_pipeline,
                                 label_names: List[str], n_repeats: int = 15,
                                 seed: int = RANDOM_SEED) -> pd.Series:
    """
    Model-agnostic "native" importance, computed the rigorous way: for
    every Leave-One-Subject-Out fold, fit a fresh clone of `model_pipeline`
    on the 14 training subjects and compute permutation importance
    (drop in macro-F1 when a feature is shuffled) on the *held-out*
    subject. Averaging these held-out importances across all 15 folds
    avoids the "looks overfit" distortion of computing permutation
    importance on data the model was already fit to (which is what a
    naive single fit-on-everything approach would do), and mirrors the
    same LOSO protocol already used to evaluate accuracy -- so feature
    importance and predictive performance are assessed under identical,
    consistent conditions.
    """
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["subject_id"].values
    logo = LeaveOneGroupOut()

    per_fold = []
    for train_idx, test_idx in logo.split(X, y, groups):
        model = clone(model_pipeline)
        model.fit(X[train_idx], y[train_idx])
        result = permutation_importance(model, X[test_idx], y[test_idx], n_repeats=n_repeats,
                                         random_state=seed, scoring="f1_macro", n_jobs=-1)
        per_fold.append(result.importances_mean)
    mean_importance = np.mean(per_fold, axis=0)
    return pd.Series(mean_importance, index=feature_cols).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Per-modality aggregation
# --------------------------------------------------------------------------
def aggregate_by_modality(importance: pd.Series) -> pd.Series:
    df = importance.reset_index()
    df.columns = ["feature", "importance"]
    df["modality"] = df["feature"].apply(feature_to_modality)
    agg = df.groupby("modality")["importance"].apply(lambda s: np.abs(s).sum())
    return agg.sort_values(ascending=False)


def univariate_discriminability(df: pd.DataFrame, feature_cols: List[str],
                                 label_col: str = "label") -> pd.DataFrame:
    """
    One-way ANOVA F-statistic per feature across the three classes: a
    simple, model-free measure of how strongly each *individual* feature's
    distribution differs by condition. Included as a complementary,
    non-multivariate cross-check alongside native/SHAP importance -- a
    feature can show a large univariate F-statistic (clearly different
    average value per class) while still receiving a modest SHAP/
    permutation importance if it is highly correlated with other features
    the model already uses (their contributions get "split"). Comparing
    the two views is itself a useful interpretability finding, discussed
    in docs/REPORT.md.
    """
    from scipy.stats import f_oneway
    rows = []
    groups_by_class = {c: df.loc[df[label_col] == c] for c in df[label_col].unique()}
    for f in feature_cols:
        groups = [g[f].values for g in groups_by_class.values()]
        stat, p = f_oneway(*groups)
        rows.append({"feature": f, "F_stat": stat, "p_value": p, "modality": feature_to_modality(f)})
    return pd.DataFrame(rows).sort_values("F_stat", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from utils import DATA_PROCESSED_DIR, TABLES_DIR, LABEL_NAMES, save_pickle, MODELS_DIR
    from modeling import get_feature_columns, build_models
    from plots import (plot_feature_importance_bar, plot_modality_bar, plot_shap_beeswarm)
    from sklearn.base import clone

    set_all_seeds()
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "features_normalized.csv"))
    feature_cols = get_feature_columns(df)
    X_all = df[feature_cols].values
    y_all = df["label"].values

    summary = pd.read_csv(os.path.join(TABLES_DIR, "model_comparison.csv"))
    best_model_name = summary.sort_values("f1_macro", ascending=False).iloc[0]["model"]
    print(f"Training final deployment model on ALL subjects: {best_model_name}")

    models = build_models()
    final_model = clone(models[best_model_name])
    final_model.fit(X_all, y_all)
    save_pickle(final_model, os.path.join(MODELS_DIR, "final_model.pkl"))

    # --- (a) Native importance: LOSO-aggregated permutation importance ---
    # (RandomForest's built-in impurity importance is also saved separately
    # below as a bonus cross-check, since it's cheap and commonly expected.)
    print("Computing native feature importance (permutation importance, "
          "aggregated across LOSO folds -- ~15-30s)...")
    native_imp = loso_permutation_importance(df, feature_cols, models[best_model_name], LABEL_NAMES)
    native_imp.to_csv(os.path.join(TABLES_DIR, "feature_importance_native.csv"), header=["importance"])
    plot_feature_importance_bar(list(native_imp.index), list(native_imp.values),
                                 title=f"Top features — native/permutation importance ({best_model_name})",
                                 name="feature_importance_native.png")
    modality_native = aggregate_by_modality(native_imp)
    modality_native.to_csv(os.path.join(TABLES_DIR, "modality_importance_native.csv"), header=["importance"])
    plot_modality_bar(list(modality_native.index), list(modality_native.values),
                       title="Per-modality contribution — native/permutation importance",
                       name="modality_importance_native.png")

    # Bonus cross-check: RandomForest's built-in impurity-based importance
    # (cheap, commonly expected, and a nice independent sanity check against
    # the LOSO permutation importance above).
    print("Computing bonus cross-check: RandomForest impurity-based importance...")
    rf_model = clone(models["Random Forest"])
    rf_model.fit(X_all, y_all)
    rf_imp = native_feature_importance(rf_model, X_all, y_all, feature_cols)
    rf_imp.to_csv(os.path.join(TABLES_DIR, "feature_importance_rf_impurity.csv"), header=["importance"])

    # --- (b) Shapley values (custom sampling estimator) ---
    print("Computing Shapley values (custom sampling estimator; ~1-2 min)...")
    rng = np.random.default_rng(RANDOM_SEED)
    # Stratified sample of instances to explain (keep runtime reasonable).
    sample_idx = []
    for label in LABEL_NAMES:
        idx = np.where(y_all == label)[0]
        take = min(20, len(idx))
        sample_idx.extend(rng.choice(idx, size=take, replace=False))
    sample_idx = np.array(sample_idx)
    X_explain = X_all[sample_idx]

    background_idx = rng.choice(len(X_all), size=min(40, len(X_all)), replace=False)
    background = X_all[background_idx]

    # Explain P(predicted class of x | x) so the attribution is always
    # "why did the model favor whichever class it actually predicted".
    pred_labels = final_model.predict(X_explain)
    proba_fn_all = final_model.predict_proba
    class_order = list(final_model.classes_)

    shap_matrix = np.zeros((len(X_explain), len(feature_cols)))
    for cls in LABEL_NAMES:
        mask = pred_labels == cls
        if mask.sum() == 0:
            continue
        cls_idx = class_order.index(cls)
        predict_fn = lambda Z, ci=cls_idx: proba_fn_all(Z)[:, ci]
        shap_matrix[mask] = shapley_values_sampling(predict_fn, X_explain[mask], background,
                                                     n_samples=150, rng=rng)

    # Verify efficiency property as a correctness check (report result);
    # done class-by-class since each instance's target class varies.
    all_errs = []
    for cls in LABEL_NAMES:
        mask = pred_labels == cls
        if mask.sum() == 0:
            continue
        cls_idx = class_order.index(cls)
        predict_fn = lambda Z, ci=cls_idx: proba_fn_all(Z)[:, ci]
        check = verify_shapley_efficiency(shap_matrix[mask], predict_fn, X_explain[mask], background)
        all_errs.append(check)
    mean_target = np.mean([c["mean_abs_target"] for c in all_errs])
    mean_err = np.mean([c["mean_abs_error"] for c in all_errs])
    print(f"  Shapley efficiency check: mean|error|={mean_err:.4f} vs mean|target|={mean_target:.4f} "
          f"({100 * mean_err / (mean_target + 1e-9):.1f}% relative error)")

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    shap_imp = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
    shap_imp.to_csv(os.path.join(TABLES_DIR, "feature_importance_shap.csv"), header=["mean_abs_shap"])
    plot_feature_importance_bar(list(shap_imp.index), list(shap_imp.values),
                                 title=f"Top features — mean |Shapley value| ({best_model_name})",
                                 name="feature_importance_shap.png", xlabel="Mean |Shapley value|")
    modality_shap = aggregate_by_modality(shap_imp)
    modality_shap.to_csv(os.path.join(TABLES_DIR, "modality_importance_shap.csv"), header=["importance"])
    plot_modality_bar(list(modality_shap.index), list(modality_shap.values),
                       title="Per-modality contribution — mean |Shapley value|",
                       name="modality_importance_shap.png")
    plot_shap_beeswarm(feature_cols, shap_matrix, X_explain,
                        title=f"SHAP summary (custom sampling estimator) — {best_model_name}",
                        name="shap_summary_beeswarm.png")

    print("\nTop 10 features by mean |Shapley value|:")
    print(shap_imp.head(10).to_string())
    print("\nTop modalities by Shapley-based importance:")
    print(modality_shap.to_string())

    # --- (c) Complementary univariate discriminability (ANOVA F-stat) ---
    print("\nComputing univariate ANOVA F-statistics (complementary cross-check)...")
    univ = univariate_discriminability(df, feature_cols)
    univ.to_csv(os.path.join(TABLES_DIR, "univariate_anova.csv"), index=False)
    modality_univ = univ.groupby("modality")["F_stat"].mean().sort_values(ascending=False)
    modality_univ.to_csv(os.path.join(TABLES_DIR, "modality_importance_univariate.csv"), header=["mean_F_stat"])
    from plots import plot_modality_raw_bar
    plot_modality_raw_bar(list(modality_univ.index), list(modality_univ.values),
                           title="Per-modality univariate discriminability (mean ANOVA F-stat)",
                           name="modality_importance_univariate.png")
    print("\nTop modalities by univariate ANOVA F-statistic:")
    print(modality_univ.to_string())

    print(f"\nSaved interpretability tables to {TABLES_DIR} and figures to results/figures/")
