#!/usr/bin/env python3
"""Train a single FT-Transformer configuration on California Housing.

Usage
-----
    python scripts/train.py
    python scripts/train.py --d-token 64 --n-blocks 2 --n-heads 4 --max-epochs 150
    python scripts/train.py --synthetic --max-epochs 2   # fast offline smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ft_transformer.config import ModelConfig, TrainConfig
from ft_transformer.data import (
    build_dataloaders,
    load_raw_housing_data,
    make_synthetic_housing_data,
    prepare_data,
)
from ft_transformer.engine import evaluate, run_training
from ft_transformer.model import FTTransformer
from ft_transformer.utils import count_trainable_parameters, get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of fetching California Housing.",
    )
    parser.add_argument("--checkpoint", default="model_checkpoints/model.pt")
    parser.add_argument("--out", default="results/tables/train_result.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    if args.synthetic:
        X, y, _ = make_synthetic_housing_data()
    else:
        X, y, _ = load_raw_housing_data()

    split = prepare_data(X, y)
    loaders = build_dataloaders(split, device=device)

    model_cfg = ModelConfig(
        n_features=X.shape[1],
        d_token=args.d_token,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )

    n_params = count_trainable_parameters(
        FTTransformer(
            model_cfg.n_features,
            model_cfg.d_token,
            model_cfg.n_blocks,
            model_cfg.n_heads,
            model_cfg.dropout,
        )
    )
    print(f"FTTransformer trainable parameters: {n_params:,}")

    result = run_training(loaders, model_cfg, train_cfg, device)

    tr_rmse, tr_r2, _ = evaluate(
        result.model,
        torch.tensor(loaders.train_inner_X, dtype=torch.float32),
        loaders.train_inner_y,
        device,
    )

    print(f"Best epoch : {result.best_epoch + 1}")
    print(f"Train RMSE : {tr_rmse:.4f}")
    print(f"Val   RMSE : {result.val_rmse:.4f}")
    print(f"Test  RMSE : {result.test_rmse:.4f}")
    print(f"Test  R2   : {result.test_r2:.4f}")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "best_epoch": result.best_epoch,
                "train_rmse": tr_rmse,
                "val_rmse": result.val_rmse,
                "test_rmse": result.test_rmse,
                "test_r2": result.test_r2,
                "model_config": vars(model_cfg),
                "train_config": vars(train_cfg),
            },
            indent=2,
        )
    )
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
