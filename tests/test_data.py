"""Unit tests for :mod:`ft_transformer.data`.

The default (non-network) job only exercises the synthetic-data path so the
suite runs offline. The real California Housing download is covered separately
by a test marked ``@pytest.mark.network``, run in its own CI job.
"""

from __future__ import annotations

import numpy as np
import pytest

from ft_transformer.config import DataConfig
from ft_transformer.data import (
    build_dataloaders,
    load_raw_housing_data,
    make_synthetic_housing_data,
    prepare_data,
)


def test_synthetic_data_shape():
    X, y, names = make_synthetic_housing_data(n_samples=500, n_features=8, seed=42)
    assert X.shape == (500, 8)
    assert y.shape == (500,)
    assert len(names) == 8


def test_synthetic_data_is_deterministic_given_seed():
    X1, y1, _ = make_synthetic_housing_data(seed=7)
    X2, y2, _ = make_synthetic_housing_data(seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_prepare_data_split_sizes(synthetic_xy):
    X, y, _ = synthetic_xy
    cfg = DataConfig(test_size=0.2, random_state=42)
    split = prepare_data(X, y, cfg)
    n = len(X)
    assert len(split.X_train_scaled) == n - int(n * 0.2)
    assert len(split.X_test_scaled) == int(n * 0.2)


def test_prepare_data_no_leakage(synthetic_xy):
    """The scaler must be fit on train only: test-set mean should NOT be ~0."""
    X, y, _ = synthetic_xy
    split = prepare_data(X, y, DataConfig())
    # Train features, scaled by a scaler fit on themselves, should have ~zero mean.
    assert np.allclose(split.X_train_scaled.mean(axis=0), 0.0, atol=1e-6)
    # Test features, scaled by the *train* scaler, generally will NOT have exactly
    # zero mean -- this is the leakage check.
    assert not np.allclose(split.X_test_scaled.mean(axis=0), 0.0, atol=1e-6)


def test_build_dataloaders_shapes(synthetic_split, device):
    loaders = build_dataloaders(
        synthetic_split, DataConfig(), batch_size_train=16, batch_size_eval=32, device=device
    )
    n_total_train = len(synthetic_split.X_train_scaled)
    n_val = int(n_total_train * 0.1)
    n_inner = n_total_train - n_val

    assert loaders.X_val_t.shape[0] == n_val
    assert loaders.train_inner_X.shape[0] == n_inner
    assert loaders.X_test_t.shape[0] == len(synthetic_split.X_test_scaled)


@pytest.mark.network
def test_real_dataset_loads_and_has_expected_shape():
    """Fetches the real California Housing dataset from scikit-learn's mirror.
    Skipped by default; run explicitly with `pytest -m network`."""
    X, y, names = load_raw_housing_data()
    assert X.shape == (20640, 8)
    assert y.shape == (20640,)
    assert names == [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
