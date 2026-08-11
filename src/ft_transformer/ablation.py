"""Architectural ablation studies: attention-head count and transformer depth.

Reproduces Cells 11-12 and 17 of the research notebook (Sections 5.3-5.4 of the
report): sweeping ``n_heads`` at a fixed depth, then sweeping ``n_blocks`` at the
best head count found, and finally assembling the full grid for the heatmap.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ft_transformer.config import ModelConfig, TrainConfig
from ft_transformer.data import Loaders
from ft_transformer.engine import evaluate, run_training


def ablate_attention_heads(
    loaders: Loaders,
    heads_grid: list[int],
    base_model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict[int, dict[str, float]]:
    """Sweep ``n_heads`` while holding ``d_token`` and ``n_blocks`` fixed."""
    results: dict[int, dict[str, float]] = {}
    if verbose:
        print(
            f"Ablation -- attention heads (d_token={base_model_cfg.d_token}, "
            f"n_blocks={base_model_cfg.n_blocks})"
        )
    for n_h in heads_grid:
        cfg = replace(base_model_cfg, n_heads=n_h)
        result = run_training(loaders, cfg, train_cfg, device, verbose=verbose)
        tr_rmse, _, _ = evaluate(
            result.model,
            torch.tensor(loaders.train_inner_X, dtype=torch.float32),
            loaders.train_inner_y,
            device,
        )
        results[n_h] = dict(train_rmse=tr_rmse, test_rmse=result.test_rmse, test_r2=result.test_r2)
        if verbose:
            print(
                f"  n_heads={n_h:>2}  train_rmse={tr_rmse:.4f}  "
                f"test_rmse={result.test_rmse:.4f}  test_r2={result.test_r2:.4f}"
            )
    return results


def ablate_transformer_depth(
    loaders: Loaders,
    blocks_grid: list[int],
    base_model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict[int, dict[str, float]]:
    """Sweep ``n_blocks`` while holding ``d_token`` and ``n_heads`` fixed."""
    results: dict[int, dict[str, float]] = {}
    if verbose:
        print(
            f"Ablation -- transformer depth (d_token={base_model_cfg.d_token}, "
            f"n_heads={base_model_cfg.n_heads})"
        )
    for n_b in blocks_grid:
        cfg = replace(base_model_cfg, n_blocks=n_b)
        result = run_training(loaders, cfg, train_cfg, device, verbose=verbose)
        tr_rmse, _, _ = evaluate(
            result.model,
            torch.tensor(loaders.train_inner_X, dtype=torch.float32),
            loaders.train_inner_y,
            device,
        )
        results[n_b] = dict(train_rmse=tr_rmse, test_rmse=result.test_rmse, test_r2=result.test_r2)
        if verbose:
            print(
                f"  n_blocks={n_b}  train_rmse={tr_rmse:.4f}  "
                f"test_rmse={result.test_rmse:.4f}  test_r2={result.test_r2:.4f}"
            )
    return results


def best_key_by_test_rmse(results: dict[int, dict[str, float]]) -> int:
    """Return the grid value (n_heads or n_blocks) with the lowest test RMSE."""
    return min(results, key=lambda k: results[k]["test_rmse"])


def build_full_ablation_grid(
    loaders: Loaders,
    heads_list: list[int],
    blocks_list: list[int],
    heads_results: dict[int, dict[str, float]],
    depth_results: dict[int, dict[str, float]],
    best_heads: int,
    d_token: int,
    train_cfg: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict[tuple[int, int], float]:
    """Assemble the full ``heads_list x blocks_list`` test-RMSE grid for the heatmap.

    Reuses cached results from :func:`ablate_attention_heads` (all entries at
    ``n_blocks=2``) and :func:`ablate_transformer_depth` (all entries at
    ``n_heads=best_heads``); any remaining ``(n_heads, n_blocks)`` combination not
    covered by either 1-D sweep is trained fresh. This mirrors Cell 17 of the
    research notebook exactly, including which cells are cache hits.
    """
    heatmap_data: dict[tuple[int, int], float] = {}
    for h in heads_list:
        for b in blocks_list:
            if b == 2 and h in heads_results:
                heatmap_data[(h, b)] = heads_results[h]["test_rmse"]
            elif h == best_heads and b in depth_results:
                heatmap_data[(h, b)] = depth_results[b]["test_rmse"]
            else:
                if verbose:
                    print(f"  Computing n_heads={h}, n_blocks={b} ...")
                cfg = ModelConfig(
                    n_features=loaders.train_inner_X.shape[1],
                    d_token=d_token,
                    n_blocks=b,
                    n_heads=h,
                )
                result = run_training(loaders, cfg, train_cfg, device, verbose=verbose)
                heatmap_data[(h, b)] = result.test_rmse
    return heatmap_data
