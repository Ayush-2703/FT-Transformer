"""Unit tests for :mod:`ft_transformer.visualize`.

Each test only checks that a non-empty PNG is produced (not pixel content), since
the goal here is to catch integration bugs (wrong keys, shape mismatches, an
unset backend crashing headless CI) rather than to pin down exact rendering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ft_transformer.visualize import (
    plot_ablation_heatmap,
    plot_loss_curves,
    plot_predicted_vs_actual,
    plot_rmse_comparison,
)


def _assert_nonempty_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_rmse_comparison(tmp_path):
    baseline_results = {
        "Linear Reg": {"train_rmse": 0.72, "test_rmse": 0.75},
        "DT Tuned": {"train_rmse": 0.48, "test_rmse": 0.65},
    }
    out = tmp_path / "fig1.png"
    fig = plot_rmse_comparison(
        baseline_results, ft_train_rmse=0.45, ft_test_rmse=0.50, out_path=out
    )
    _assert_nonempty_png(out)
    assert fig is not None


def test_plot_loss_curves(tmp_path):
    out = tmp_path / "fig2.png"
    train_losses = [1.0, 0.6, 0.4, 0.35, 0.3]
    val_losses = [0.9, 0.7, 0.6, 0.58, 0.57]
    plot_loss_curves(train_losses, val_losses, best_epoch=3, out_path=out)
    _assert_nonempty_png(out)


def test_plot_ablation_heatmap(tmp_path):
    out = tmp_path / "fig3.png"
    heads_list = [1, 4, 8]
    blocks_list = [1, 2, 3]
    heatmap_data = {(h, b): 0.5 + 0.01 * h + 0.01 * b for h in heads_list for b in blocks_list}
    plot_ablation_heatmap(heads_list, blocks_list, heatmap_data, out_path=out)
    _assert_nonempty_png(out)


def test_plot_predicted_vs_actual(tmp_path):
    out = tmp_path / "fig4.png"
    rng = np.random.default_rng(0)
    actual = rng.normal(size=100)
    preds = actual + rng.normal(scale=0.1, size=100)
    plot_predicted_vs_actual(actual, preds, rmse=0.1, r2=0.95, out_path=out)
    _assert_nonempty_png(out)
