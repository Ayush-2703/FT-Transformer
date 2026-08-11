#!/usr/bin/env python3
"""Run the attention-head and transformer-depth ablation studies.

Usage
-----
    python scripts/run_ablation.py
    python scripts/run_ablation.py --synthetic --max-epochs 5   # fast smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ft_transformer.ablation import (
    ablate_attention_heads,
    ablate_transformer_depth,
    best_key_by_test_rmse,
)
from ft_transformer.config import ModelConfig, TrainConfig
from ft_transformer.data import (
    build_dataloaders,
    load_raw_housing_data,
    make_synthetic_housing_data,
    prepare_data,
)
from ft_transformer.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads-grid", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--blocks-grid", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", default="results/tables/ablation_results.json")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    X, y, _ = make_synthetic_housing_data() if args.synthetic else load_raw_housing_data()
    split = prepare_data(X, y)
    loaders = build_dataloaders(split, device=device)

    base_cfg = ModelConfig(n_features=X.shape[1], d_token=args.d_token, n_blocks=2, n_heads=4)
    train_cfg = TrainConfig(max_epochs=args.max_epochs, patience=args.patience)

    heads_results = ablate_attention_heads(loaders, args.heads_grid, base_cfg, train_cfg, device)
    best_heads = best_key_by_test_rmse(heads_results)
    print(f"\nBest n_heads: {best_heads}")

    depth_cfg = ModelConfig(
        n_features=X.shape[1], d_token=args.d_token, n_blocks=2, n_heads=best_heads
    )
    depth_results = ablate_transformer_depth(
        loaders, args.blocks_grid, depth_cfg, train_cfg, device
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "heads_ablation": heads_results,
                "depth_ablation": depth_results,
                "best_heads": best_heads,
            },
            indent=2,
        )
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
