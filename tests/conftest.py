"""Shared pytest fixtures.

Every fixture here uses `make_synthetic_housing_data` rather than fetching the
real California Housing dataset, so the entire non-network test suite runs
fast and deterministically offline (see `pyproject.toml`'s
`addopts = "-m 'not network'"`).
"""

from __future__ import annotations

import pytest
import torch

from ft_transformer.config import DataConfig, ModelConfig, TrainConfig
from ft_transformer.data import build_dataloaders, make_synthetic_housing_data, prepare_data
from ft_transformer.utils import set_seed


@pytest.fixture(autouse=True)
def _fixed_seed():
    """Fix every RNG before each test for full determinism."""
    set_seed(42)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def synthetic_xy():
    return make_synthetic_housing_data(n_samples=400, n_features=8, seed=42)


@pytest.fixture
def synthetic_split(synthetic_xy):
    X, y, _ = synthetic_xy
    return prepare_data(X, y, DataConfig())


@pytest.fixture
def synthetic_loaders(synthetic_split, device):
    return build_dataloaders(
        synthetic_split, DataConfig(), batch_size_train=32, batch_size_eval=64, device=device
    )


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    """A small, fast-to-train architecture for unit tests."""
    return ModelConfig(n_features=8, d_token=16, n_blocks=1, n_heads=2, dropout=0.0)


@pytest.fixture
def tiny_train_cfg() -> TrainConfig:
    """Few epochs, tiny patience -- just enough to exercise the training loop."""
    return TrainConfig(lr=1e-3, weight_decay=1e-4, max_epochs=3, patience=2, seed=42)
