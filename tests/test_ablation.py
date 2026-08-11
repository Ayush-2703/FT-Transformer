"""Unit tests for :mod:`ft_transformer.ablation`."""

from __future__ import annotations

from ft_transformer.ablation import (
    ablate_attention_heads,
    ablate_transformer_depth,
    best_key_by_test_rmse,
    build_full_ablation_grid,
)
from ft_transformer.config import ModelConfig, TrainConfig


def test_ablate_attention_heads_covers_full_grid(synthetic_loaders, device):
    base_cfg = ModelConfig(n_features=8, d_token=16, n_blocks=1, n_heads=2)
    train_cfg = TrainConfig(max_epochs=2, patience=1, seed=42)
    results = ablate_attention_heads(
        synthetic_loaders, [1, 2, 4], base_cfg, train_cfg, device, verbose=False
    )
    assert set(results.keys()) == {1, 2, 4}
    for res in results.values():
        assert {"train_rmse", "test_rmse", "test_r2"} <= res.keys()
        assert res["test_rmse"] >= 0


def test_ablate_transformer_depth_covers_full_grid(synthetic_loaders, device):
    base_cfg = ModelConfig(n_features=8, d_token=16, n_blocks=1, n_heads=2)
    train_cfg = TrainConfig(max_epochs=2, patience=1, seed=42)
    results = ablate_transformer_depth(
        synthetic_loaders, [1, 2], base_cfg, train_cfg, device, verbose=False
    )
    assert set(results.keys()) == {1, 2}


def test_best_key_by_test_rmse_picks_the_minimum():
    fake_results = {
        1: {"test_rmse": 0.9},
        4: {"test_rmse": 0.5},
        8: {"test_rmse": 0.6},
    }
    assert best_key_by_test_rmse(fake_results) == 4


def test_build_full_ablation_grid_covers_every_cell_and_reuses_cache(synthetic_loaders, device):
    heads_list = [1, 2]
    blocks_list = [1, 2]
    train_cfg = TrainConfig(max_epochs=2, patience=1, seed=42)

    # Cached from a heads sweep at n_blocks=2, and a depth sweep at n_heads=1 (best_heads).
    heads_results = {1: {"test_rmse": 0.111}, 2: {"test_rmse": 0.222}}
    depth_results = {1: {"test_rmse": 0.333}, 2: {"test_rmse": 0.111}}

    grid = build_full_ablation_grid(
        synthetic_loaders,
        heads_list,
        blocks_list,
        heads_results,
        depth_results,
        best_heads=1,
        d_token=16,
        train_cfg=train_cfg,
        device=device,
        verbose=False,
    )

    # Every (h, b) combination must be present.
    assert set(grid.keys()) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    # Cache hits must be reused verbatim (not recomputed).
    assert grid[(1, 2)] == 0.111  # from heads_results
    assert grid[(1, 1)] == 0.333  # from depth_results (h == best_heads)
    # (2, 1) is neither b==2 nor h==best_heads, so it must have been freshly trained.
    assert grid[(2, 1)] not in (0.111, 0.222, 0.333)
