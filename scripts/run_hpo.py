#!/usr/bin/env python3
"""Run the 50-trial Optuna TPE hyperparameter search, then retrain the best
configuration across three seeds to quantify result variance.

Usage
-----
    python scripts/run_hpo.py --n-trials 50
    python scripts/run_hpo.py --synthetic --n-trials 3 --max-epochs 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ft_transformer.config import TrainConfig
from ft_transformer.data import (
    build_dataloaders,
    load_raw_housing_data,
    make_synthetic_housing_data,
    prepare_data,
)
from ft_transformer.engine import evaluate, run_training
from ft_transformer.hpo import best_config_from_study, run_search
from ft_transformer.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--final-max-epochs", type=int, default=150)
    parser.add_argument("--final-patience", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint", default="model_checkpoints/model_optuna_best.pt")
    parser.add_argument("--out", default="results/tables/hpo_results.json")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    X, y, _ = make_synthetic_housing_data() if args.synthetic else load_raw_housing_data()
    split = prepare_data(X, y)
    loaders = build_dataloaders(split, device=device)

    study = run_search(
        loaders,
        X.shape[1],
        device,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(f"Best params   : {study.best_params}")
    print(f"Best val RMSE : {study.best_value:.4f}")

    cfg = best_config_from_study(study, X.shape[1])
    model_cfg = cfg["model"]

    final_runs = []
    best_model = None
    best_te_rmse = float("inf")
    for seed in args.seeds:
        train_cfg = TrainConfig(
            **cfg["train_partial"],
            max_epochs=args.final_max_epochs,
            patience=args.final_patience,
            seed=seed,
        )
        result = run_training(loaders, model_cfg, train_cfg, device)
        tr_rmse, _, _ = evaluate(
            result.model,
            torch.tensor(loaders.train_inner_X, dtype=torch.float32),
            loaders.train_inner_y,
            device,
        )
        print(
            f"  seed={seed:>3}: train RMSE={tr_rmse:.4f}  test RMSE={result.test_rmse:.4f}  "
            f"test R2={result.test_r2:.4f}"
        )
        final_runs.append(
            {
                "seed": seed,
                "train_rmse": tr_rmse,
                "test_rmse": result.test_rmse,
                "test_r2": result.test_r2,
            }
        )
        if result.test_rmse < best_te_rmse:
            best_te_rmse = result.test_rmse
            best_model = result.model

    rmse_vals = [r["test_rmse"] for r in final_runs]
    r2_vals = [r["test_r2"] for r in final_runs]
    print(f"Final Test RMSE : {np.mean(rmse_vals):.4f}  +/-  {np.std(rmse_vals):.4f}")
    print(f"Final Test R2   : {np.mean(r2_vals):.4f}  +/-  {np.std(r2_vals):.4f}")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "best_params": study.best_params,
                "best_val_rmse": study.best_value,
                "final_runs": final_runs,
                "test_rmse_mean": float(np.mean(rmse_vals)),
                "test_rmse_std": float(np.std(rmse_vals)),
                "test_r2_mean": float(np.mean(r2_vals)),
                "test_r2_std": float(np.std(r2_vals)),
            },
            indent=2,
        )
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
