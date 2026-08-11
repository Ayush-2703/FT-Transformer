#!/usr/bin/env python3
"""Run the entire experimental pipeline end-to-end: baselines, default FT-Transformer
training, ablations, Optuna search, final 3-seed evaluation, and all four figures.

This reproduces the complete research notebook as a single reusable script.

Usage
-----
    python scripts/run_full_pipeline.py                       # full reproduction (slow)
    python scripts/run_full_pipeline.py --synthetic --fast    # CI-friendly smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ft_transformer.ablation import (
    ablate_attention_heads,
    ablate_transformer_depth,
    best_key_by_test_rmse,
    build_full_ablation_grid,
)
from ft_transformer.baselines import format_results_table, run_classical_baselines
from ft_transformer.config import ModelConfig, TrainConfig
from ft_transformer.data import (
    build_dataloaders,
    load_raw_housing_data,
    make_synthetic_housing_data,
    prepare_data,
)
from ft_transformer.engine import evaluate, run_training
from ft_transformer.hpo import best_config_from_study, run_search
from ft_transformer.utils import get_device, set_seed
from ft_transformer.visualize import (
    plot_ablation_heatmap,
    plot_loss_curves,
    plot_predicted_vs_actual,
    plot_rmse_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Drastically shrink epochs/trials for a quick smoke test.",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    max_epochs = 5 if args.fast else 150
    patience = 3 if args.fast else 15
    hpo_epochs = 3 if args.fast else 50
    hpo_patience = 2 if args.fast else 10
    n_trials = 3 if args.fast else 50
    heads_grid = [1, 4] if args.fast else [1, 4, 8]
    blocks_grid = [1, 2] if args.fast else [1, 2, 3]

    results_dir = Path(args.results_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}\n")

    # -- Data -------------------------------------------------------------
    X, y, _ = make_synthetic_housing_data() if args.synthetic else load_raw_housing_data()
    split = prepare_data(X, y)
    loaders = build_dataloaders(split, device=device)
    n_features = X.shape[1]

    # -- Classical baselines ------------------------------------------------
    print("=" * 70)
    print("Classical baselines")
    print("=" * 70)
    baseline_results = run_classical_baselines(
        split.X_train_scaled, split.y_train, split.X_test_scaled, split.y_test
    )
    print(format_results_table(baseline_results))

    # -- Default FT-Transformer ---------------------------------------------
    print("\n" + "=" * 70)
    print("Default FT-Transformer")
    print("=" * 70)
    default_model_cfg = ModelConfig(n_features=n_features)
    default_train_cfg = TrainConfig(max_epochs=max_epochs, patience=patience)
    default_result = run_training(loaders, default_model_cfg, default_train_cfg, device)
    tr_rmse_default, _, _ = evaluate(
        default_result.model,
        torch.tensor(loaders.train_inner_X, dtype=torch.float32),
        loaders.train_inner_y,
        device,
    )
    print(f"Test RMSE: {default_result.test_rmse:.4f}  Test R2: {default_result.test_r2:.4f}")

    # -- Ablations ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Ablations")
    print("=" * 70)
    ablation_train_cfg = TrainConfig(max_epochs=max_epochs, patience=patience)
    heads_results = ablate_attention_heads(
        loaders, heads_grid, default_model_cfg, ablation_train_cfg, device
    )
    best_heads = best_key_by_test_rmse(heads_results)
    depth_cfg = ModelConfig(n_features=n_features, n_heads=best_heads)
    depth_results = ablate_transformer_depth(
        loaders, blocks_grid, depth_cfg, ablation_train_cfg, device
    )

    # -- Optuna search + final 3-seed evaluation ------------------------------
    print("\n" + "=" * 70)
    print("Optuna hyperparameter search")
    print("=" * 70)
    study = run_search(
        loaders,
        n_features,
        device,
        n_trials=n_trials,
        max_epochs=hpo_epochs,
        patience=hpo_patience,
        show_progress_bar=False,
    )
    cfg = best_config_from_study(study, n_features)
    final_runs = []
    for seed in [0, 42, 123]:
        train_cfg = TrainConfig(
            **cfg["train_partial"], max_epochs=max_epochs, patience=patience, seed=seed
        )
        result = run_training(loaders, cfg["model"], train_cfg, device, verbose=False)
        final_runs.append({"seed": seed, "test_rmse": result.test_rmse, "test_r2": result.test_r2})
        print(f"  seed={seed}: test RMSE={result.test_rmse:.4f}  test R2={result.test_r2:.4f}")
    rmse_vals = [r["test_rmse"] for r in final_runs]
    r2_vals = [r["test_r2"] for r in final_runs]

    # -- Figures ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Figures")
    print("=" * 70)
    plot_rmse_comparison(
        baseline_results,
        tr_rmse_default,
        default_result.test_rmse,
        results_dir / "figures" / "figure1_rmse_comparison.png",
    )
    plot_loss_curves(
        default_result.train_losses,
        default_result.val_losses,
        default_result.best_epoch,
        results_dir / "figures" / "figure2_loss_curves.png",
    )

    heatmap_data = build_full_ablation_grid(
        loaders,
        heads_grid,
        blocks_grid,
        heads_results,
        depth_results,
        best_heads,
        default_model_cfg.d_token,
        ablation_train_cfg,
        device,
        verbose=True,
    )
    plot_ablation_heatmap(
        heads_grid,
        blocks_grid,
        heatmap_data,
        results_dir / "figures" / "figure3_ablation_heatmap.png",
    )

    plot_predicted_vs_actual(
        loaders.y_test_np,
        default_result.preds,
        default_result.test_rmse,
        default_result.test_r2,
        results_dir / "figures" / "figure4_pred_vs_actual.png",
    )
    print(f"Saved 4 figures under {results_dir / 'figures'}")

    # -- Summary --------------------------------------------------------------
    summary = {
        "baselines": baseline_results,
        "ft_transformer_default": {
            "train_rmse": tr_rmse_default,
            "test_rmse": default_result.test_rmse,
            "test_r2": default_result.test_r2,
        },
        "ablation_heads": heads_results,
        "ablation_depth": depth_results,
        "optuna_best_params": study.best_params,
        "optuna_best_val_rmse": study.best_value,
        "final_3seed_runs": final_runs,
        "final_test_rmse_mean": float(np.mean(rmse_vals)),
        "final_test_rmse_std": float(np.std(rmse_vals)),
        "final_test_r2_mean": float(np.mean(r2_vals)),
        "final_test_r2_std": float(np.std(r2_vals)),
    }
    out_path = results_dir / "tables" / "full_pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {out_path}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
