"""Unit tests for :mod:`ft_transformer.engine` (training loop and evaluation)."""

from __future__ import annotations

import numpy as np
import torch

from ft_transformer.engine import evaluate, run_training, train_epoch
from ft_transformer.model import FTTransformer


def test_train_epoch_reduces_loss_over_several_calls(synthetic_loaders, device):
    model = FTTransformer(n_features=8, d_token=16, n_blocks=1, n_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    losses = []
    for _ in range(5):
        loss = train_epoch(model, synthetic_loaders.train_loader, optimizer, criterion, device)
        losses.append(loss)

    assert all(np.isfinite(losses))
    # Loss should trend downward over 5 epochs on this easy synthetic problem.
    assert losses[-1] < losses[0]


def test_evaluate_returns_expected_shapes_and_finite_metrics(synthetic_loaders, device):
    model = FTTransformer(n_features=8, d_token=16, n_blocks=1, n_heads=2)
    rmse, r2, preds = evaluate(
        model, synthetic_loaders.X_test_t, synthetic_loaders.y_test_np, device
    )

    assert isinstance(rmse, float) and np.isfinite(rmse) and rmse >= 0
    assert isinstance(r2, float) and np.isfinite(r2)
    assert preds.shape == synthetic_loaders.y_test_np.shape


def test_run_training_end_to_end(synthetic_loaders, tiny_model_cfg, tiny_train_cfg, device):
    result = run_training(synthetic_loaders, tiny_model_cfg, tiny_train_cfg, device, verbose=False)

    assert result.test_rmse >= 0
    assert -10 <= result.test_r2 <= 1.0  # R2 can be very negative for undertrained models
    assert len(result.train_losses) == len(result.val_losses)
    assert len(result.train_losses) >= 1
    assert result.preds.shape == synthetic_loaders.y_test_np.shape
    # best_epoch should be the epoch of minimum val_losses
    assert result.val_losses[result.best_epoch] == min(result.val_losses)


def test_run_training_is_reproducible_given_same_seed(
    synthetic_loaders, tiny_model_cfg, tiny_train_cfg, device
):
    result_a = run_training(
        synthetic_loaders, tiny_model_cfg, tiny_train_cfg, device, verbose=False
    )
    result_b = run_training(
        synthetic_loaders, tiny_model_cfg, tiny_train_cfg, device, verbose=False
    )
    assert result_a.test_rmse == result_b.test_rmse
    assert np.allclose(result_a.preds, result_b.preds)


def test_early_stopping_respects_patience(synthetic_loaders, tiny_model_cfg, device):
    from ft_transformer.config import TrainConfig

    train_cfg = TrainConfig(max_epochs=50, patience=1, seed=42)
    result = run_training(synthetic_loaders, tiny_model_cfg, train_cfg, device, verbose=False)
    # Training must stop no later than best_epoch + patience + 1 epochs.
    assert len(result.train_losses) <= result.best_epoch + train_cfg.patience + 1
