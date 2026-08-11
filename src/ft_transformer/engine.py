"""Training loop, evaluation, and the top-level `run_training` orchestrator.

Implements AdamW optimisation, cosine-annealing LR scheduling, gradient-norm
clipping, and early stopping on validation RMSE with best-checkpoint restoration
-- exactly as specified in Section 4.4 of the research report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ft_transformer.config import ModelConfig, TrainConfig
from ft_transformer.data import Loaders
from ft_transformer.model import FTTransformer


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> float:
    """Run one full training epoch. Returns the sample-weighted mean MSE loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        # Clip gradient norm to prevent exploding gradients at higher LRs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module, X_tensor: torch.Tensor, y_numpy: np.ndarray, device: torch.device
) -> tuple[float, float, np.ndarray]:
    """Inference-mode evaluation. Returns ``(rmse, r2, predictions)``."""
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device)).cpu().numpy()
    rmse = float(np.sqrt(np.mean((preds - y_numpy) ** 2)))
    ss_res = np.sum((preds - y_numpy) ** 2)
    ss_tot = np.sum((y_numpy - y_numpy.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return rmse, r2, preds


@dataclass
class TrainingResult:
    """Everything produced by one call to :func:`run_training`."""

    model: FTTransformer
    test_rmse: float
    test_r2: float
    val_rmse: float
    best_epoch: int
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    preds: np.ndarray | None = None


def run_training(
    loaders: Loaders,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Build an FT-Transformer and run the full AdamW + cosine-annealing training loop.

    Restores the best (lowest validation-RMSE) checkpoint before the final test-set
    evaluation, exactly mirroring Cell 9 of the research notebook.
    """
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    device = device or torch.device("cpu")

    torch.manual_seed(train_cfg.seed)  # per-run reproducibility

    model = FTTransformer(
        model_cfg.n_features,
        model_cfg.d_token,
        model_cfg.n_blocks,
        model_cfg.n_heads,
        model_cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.max_epochs)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(train_cfg.max_epochs):
        train_loss = train_epoch(
            model,
            loaders.train_loader,
            optimizer,
            criterion,
            device,
            train_cfg.grad_clip_norm,
        )
        val_rmse, _, _ = evaluate(model, loaders.X_val_t, loaders.y_val_np, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= train_cfg.patience:
            if verbose:
                print(f"  Early stop at epoch {epoch + 1} (best epoch {best_epoch + 1})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_rmse, test_r2, preds = evaluate(model, loaders.X_test_t, loaders.y_test_np, device)

    return TrainingResult(
        model=model,
        test_rmse=test_rmse,
        test_r2=test_r2,
        val_rmse=best_val_rmse,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        preds=preds,
    )
