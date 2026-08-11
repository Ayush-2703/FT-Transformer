"""Figure generation: RMSE bar chart, loss curves, ablation heatmap, pred-vs-actual.

Reproduces Cells 15, 16, 17, and 18 of the research notebook. Every function
takes plain data (dicts / arrays) rather than notebook globals, saves a PNG to
the given output path, and returns the ``matplotlib.figure.Figure`` for further
use (e.g. embedding in a report).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe backend for CI / scripts
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rmse_comparison(
    baseline_results: dict[str, dict[str, Any]],
    ft_train_rmse: float,
    ft_test_rmse: float,
    out_path: str | Path,
) -> plt.Figure:
    """Grouped bar chart: Train vs Test RMSE, classical baselines vs FT-Transformer."""
    model_names = list(baseline_results.keys()) + ["FT-Transformer"]
    train_rmses = [baseline_results[m]["train_rmse"] for m in baseline_results] + [ft_train_rmse]
    test_rmses = [baseline_results[m]["test_rmse"] for m in baseline_results] + [ft_test_rmse]

    n_models = len(model_names)
    n_classical = n_models - 1
    x = np.arange(n_models)
    w = 0.36

    train_colors = ["#FFCC80"] * n_classical + ["#64B5F6"]
    test_colors = ["#F57C00"] * n_classical + ["#1565C0"]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_tr = ax.bar(
        x - w / 2,
        train_rmses,
        w,
        color=train_colors,
        edgecolor="k",
        linewidth=0.6,
        label="Train RMSE",
    )
    bars_te = ax.bar(
        x + w / 2, test_rmses, w, color=test_colors, edgecolor="k", linewidth=0.6, label="Test RMSE"
    )

    for bar in list(bars_tr) + list(bars_te):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.006,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=18, ha="right", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(
        "Train vs Test RMSE -- Classical Baselines vs FT-Transformer",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(train_rmses), max(test_rmses)) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(fig)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
    out_path: str | Path,
) -> plt.Figure:
    """Training MSE loss and validation RMSE per epoch, with best epoch marked."""
    epochs_arr = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_arr, train_losses, label="Train MSE Loss", color="steelblue", linewidth=1.8)
    ax.plot(epochs_arr, val_losses, label="Val RMSE", color="darkorange", linewidth=1.8)
    ax.axvline(
        x=best_epoch + 1,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Best epoch ({best_epoch + 1})",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss / RMSE", fontsize=12)
    ax.set_title(
        "FT-Transformer Training and Validation Loss Curves", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    sns.despine(fig)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_ablation_heatmap(
    heads_list: list[int],
    blocks_list: list[int],
    heatmap_data: dict[tuple[int, int], float],
    out_path: str | Path,
) -> plt.Figure:
    """Seaborn heatmap of test RMSE across (n_heads x n_blocks) combinations."""
    mat = np.array([[heatmap_data[(h, b)] for b in blocks_list] for h in heads_list])
    df_heat = pd.DataFrame(
        mat,
        index=[f"heads={h}" for h in heads_list],
        columns=[f"blocks={b}" for b in blocks_list],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".4f",
        cmap="coolwarm_r",
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 12},
    )
    ax.set_title(
        "Ablation Study -- Test RMSE by Attention Heads and Depth", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Number of Transformer Blocks", fontsize=11)
    ax.set_ylabel("Number of Attention Heads", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_predicted_vs_actual(
    actual: np.ndarray,
    preds: np.ndarray,
    rmse: float,
    r2: float,
    out_path: str | Path,
) -> plt.Figure:
    """Scatter plot of predictions vs ground truth with a y=x reference line."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(actual, preds, alpha=0.3, s=9, color="steelblue", label="Predictions")

    lo = min(float(actual.min()), float(preds.min()))
    hi = max(float(actual.max()), float(preds.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="y = x (perfect)")

    textstr = f"RMSE = {rmse:.4f}\nR\u00b2   = {r2:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.65)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel("Actual House Value", fontsize=12)
    ax.set_ylabel("Predicted House Value", fontsize=12)
    ax.set_title(
        "FT-Transformer: Predicted vs Actual House Values (Test Set)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    sns.despine(fig)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig
