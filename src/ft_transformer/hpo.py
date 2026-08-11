"""Optuna-based hyperparameter search over the full FT-Transformer search space.

Reproduces Cell 13 of the research notebook: a TPE-sampled study over
``d_token``, ``n_blocks``, ``n_heads``, ``dropout``, ``lr``, and ``weight_decay``,
minimising validation RMSE, with trials that violate the
``d_token % n_heads == 0`` constraint pruned immediately.
"""

from __future__ import annotations

from typing import Any

import optuna
import torch

from ft_transformer.config import ModelConfig, TrainConfig
from ft_transformer.data import Loaders
from ft_transformer.engine import run_training

SEARCH_SPACE = {
    "d_token": [32, 64, 128, 256],
    "n_blocks": (1, 4),
    "n_heads": [1, 2, 4, 8],
    "dropout": (0.0, 0.3),
    "lr": (1e-4, 1e-2),
    "wd": (1e-6, 1e-3),
}


def build_objective(
    loaders: Loaders,
    n_features: int,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
):
    """Build an Optuna objective closure bound to a fixed dataset and device."""

    def objective(trial: optuna.Trial) -> float:
        d_token = trial.suggest_categorical("d_token", SEARCH_SPACE["d_token"])
        n_blocks = trial.suggest_int("n_blocks", *SEARCH_SPACE["n_blocks"])
        n_heads = trial.suggest_categorical("n_heads", SEARCH_SPACE["n_heads"])
        # Each head must see an integer slice of d_token.
        if d_token % n_heads != 0:
            raise optuna.TrialPruned()
        dropout = trial.suggest_float("dropout", *SEARCH_SPACE["dropout"])
        lr = trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True)
        wd = trial.suggest_float("wd", *SEARCH_SPACE["wd"], log=True)

        model_cfg = ModelConfig(
            n_features=n_features,
            d_token=d_token,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        train_cfg = TrainConfig(lr=lr, weight_decay=wd, max_epochs=max_epochs, patience=patience)
        result = run_training(loaders, model_cfg, train_cfg, device, verbose=False)
        return result.val_rmse

    return objective


def run_search(
    loaders: Loaders,
    n_features: int,
    device: torch.device,
    n_trials: int = 50,
    seed: int = 42,
    max_epochs: int = 50,
    patience: int = 10,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """Run the full TPE hyperparameter search and return the completed study."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective = build_objective(loaders, n_features, device, max_epochs, patience)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


def best_config_from_study(study: optuna.Study, n_features: int) -> dict[str, Any]:
    """Convert ``study.best_params`` into ready-to-use Model/Train config kwargs."""
    bp = study.best_params
    return {
        "model": ModelConfig(
            n_features=n_features,
            d_token=bp["d_token"],
            n_blocks=bp["n_blocks"],
            n_heads=bp["n_heads"],
            dropout=bp["dropout"],
        ),
        "train_partial": {"lr": bp["lr"], "weight_decay": bp["wd"]},
    }
