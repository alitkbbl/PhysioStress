"""
plots.py
--------
Small shared plotting helpers so every figure in results/figures/ uses a
consistent, presentable style. Kept deliberately simple (matplotlib +
seaborn only -- no extra dependencies).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import FIGURES_DIR

sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
PALETTE = {"baseline": "#4C72B0", "stress": "#C44E52", "amusement": "#55A868"}
MODEL_COLOR = "#4C72B0"


def _savefig(fig, name: str) -> str:
    path = os.path.join(FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame, name: str = "class_distribution.png") -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().reindex(["baseline", "stress", "amusement"])
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center", fontsize=11)
    ax.set_ylabel("Number of 60s windows")
    ax.set_title("Class distribution (all subjects)")
    return _savefig(fig, name)


def plot_model_comparison(summary_df: pd.DataFrame, name: str = "model_comparison.png") -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    d = summary_df.sort_values("f1_macro", ascending=True)
    y = np.arange(len(d))
    ax.barh(y - 0.18, d["accuracy"], height=0.36, label="Accuracy", color="#4C72B0")
    ax.barh(y + 0.18, d["f1_macro"], height=0.36, label="F1-macro", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.set_xlabel("Score (LOSO cross-validated)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Model comparison — Leave-One-Subject-Out CV")
    ax.legend(loc="lower right")
    for i, (acc, f1) in enumerate(zip(d["accuracy"], d["f1_macro"])):
        ax.text(acc + 0.01, i - 0.18, f"{acc:.2f}", va="center", fontsize=10)
        ax.text(f1 + 0.01, i + 0.18, f"{f1:.2f}", va="center", fontsize=10)
    return _savefig(fig, name)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str,
                           name: str = "confusion_matrix.png") -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Row-normalized proportion"}, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    return _savefig(fig, name)


def plot_per_subject_accuracy(fold_df: pd.DataFrame, model_name: str,
                               name: str = "loso_per_subject_accuracy.png") -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = fold_df.sort_values("accuracy")
    colors = ["#C44E52" if a < d["accuracy"].median() else "#55A868" for a in d["accuracy"]]
    ax.bar(d["subject_id"], d["accuracy"], color=colors)
    ax.axhline(d["accuracy"].mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean = {d['accuracy'].mean():.2f}")
    ax.set_ylabel("Held-out accuracy")
    ax.set_xlabel("Held-out subject (LOSO fold)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-subject LOSO accuracy — {model_name}")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _savefig(fig, name)


def plot_feature_importance_bar(names: List[str], values: List[float], title: str,
                                 name: str, xlabel: str = "Importance", top_n: int = 20) -> str:
    order = np.argsort(values)[::-1][:top_n]
    names_o = [names[i] for i in order][::-1]
    values_o = [values[i] for i in order][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(names_o))))
    ax.barh(names_o, values_o, color="#4C72B0")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _savefig(fig, name)


def plot_modality_bar(modality_names: List[str], values: List[float], title: str,
                       name: str) -> str:
    order = np.argsort(values)[::-1]
    names_o = [modality_names[i] for i in order]
    values_o = np.array([values[i] for i in order])
    pct = 100 * values_o / values_o.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names_o, pct, color=sns.color_palette("viridis", len(names_o)))
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.5, f"{p:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Share of total importance (%)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _savefig(fig, name)


def plot_modality_raw_bar(modality_names: List[str], values: List[float], title: str,
                           name: str, xlabel: str = "Mean ANOVA F-statistic") -> str:
    order = np.argsort(values)[::-1]
    names_o = [modality_names[i] for i in order]
    values_o = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names_o, values_o, color=sns.color_palette("viridis", len(names_o)))
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _savefig(fig, name)


def plot_shap_beeswarm(feature_names: List[str], shap_matrix: np.ndarray, feature_matrix: np.ndarray,
                        title: str, name: str, top_n: int = 15) -> str:
    """
    Manual beeswarm-style summary plot (mimics shap.summary_plot), built
    directly from this project's own Shapley-value estimator: one row per
    feature, one dot per instance, x = SHAP value, color = that instance's
    (normalized) feature value.
    """
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(order))))
    cmap = plt.get_cmap("coolwarm")
    for row_i, feat_i in enumerate(order[::-1]):
        vals = shap_matrix[:, feat_i]
        fvals = feature_matrix[:, feat_i]
        fmin, fmax = np.percentile(fvals, 2), np.percentile(fvals, 98)
        norm = np.clip((fvals - fmin) / (fmax - fmin + 1e-9), 0, 1)
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.6
        ax.scatter(vals, np.full_like(vals, row_i) + jitter, c=cmap(norm), s=18,
                   alpha=0.75, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Shapley value (impact on model output)")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Feature value (low → high)")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    return _savefig(fig, name)
