"""Data loading, preprocessing, and DataLoader construction.

Mirrors the exact preprocessing pipeline used in the research notebook
(Cells 2 and 7): an 80/20 train-test split, a `StandardScaler` fit on the
training fold only (no leakage), and a further 90/10 inner train/validation
split of the training fold for early stopping.

Also provides :func:`make_synthetic_housing_data`, which generates data with the
same shape (8 features, continuous target) without any network access. This is
used by the unit test suite and by CI's default (non-network) job, so the model
and training-loop logic can be verified quickly and deterministically anywhere,
including offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ft_transformer.config import DataConfig

N_FEATURES = 8
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@dataclass
class SplitData:
    """Container for every array/tensor produced by :func:`prepare_data`."""

    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]


def load_raw_housing_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch the California Housing dataset (8 features, n=20,640) via scikit-learn.

    Requires network access on first call (scikit-learn caches the download under
    ``~/scikit_learn_data``). For offline development and tests, use
    :func:`make_synthetic_housing_data` instead.
    """
    housing = fetch_california_housing()
    return housing.data, housing.target, list(housing.feature_names)


def make_synthetic_housing_data(
    n_samples: int = 2000, n_features: int = N_FEATURES, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic tabular regression data with California-Housing shape.

    The target is a noisy nonlinear combination of the features so that a
    transformer with feature interactions has something non-trivial to learn,
    without requiring any network access. Used by the test suite and for fast
    offline smoke-testing of the training pipeline.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    interaction = X[:, 0] * X[:, min(1, n_features - 1)]
    noise = rng.normal(scale=0.3, size=n_samples)
    y = X @ weights + 0.5 * interaction + noise
    names = [f"synthetic_feature_{i}" for i in range(n_features)]
    return X.astype(np.float32), y.astype(np.float32), names


def prepare_data(X: np.ndarray, y: np.ndarray, cfg: DataConfig | None = None) -> SplitData:
    """Run the 80/20 split + leakage-free StandardScaler pipeline.

    Parameters
    ----------
    X, y:
        Raw feature matrix and target vector.
    cfg:
        Split hyperparameters; defaults to :class:`~ft_transformer.config.DataConfig`.
    """
    cfg = cfg or DataConfig()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
    X_test_scaled = scaler.transform(X_test)  # transform only -- no leakage

    return SplitData(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=[],
    )


@dataclass
class Loaders:
    """Every DataLoader / tensor needed by the training loop."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    X_val_t: torch.Tensor
    y_val_np: np.ndarray
    X_test_t: torch.Tensor
    y_test_np: np.ndarray
    train_inner_X: np.ndarray
    train_inner_y: np.ndarray


def build_dataloaders(
    split: SplitData,
    cfg: DataConfig | None = None,
    batch_size_train: int = 256,
    batch_size_eval: int = 512,
    device: torch.device | None = None,
) -> Loaders:
    """Carve the inner 90/10 train/validation split and wrap everything in DataLoaders.

    Mirrors Cell 7 of the research notebook exactly, including the fixed
    ``torch.Generator(seed=42)`` used for the reproducible inner split.
    """
    cfg = cfg or DataConfig()
    device = device or torch.device("cpu")

    X_train_t = torch.tensor(split.X_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(split.X_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(split.y_train, dtype=torch.float32)
    y_test_t = torch.tensor(split.y_test, dtype=torch.float32)

    n_total = len(X_train_t)
    n_val = int(n_total * cfg.inner_val_size)
    n_tr_inner = n_total - n_val

    gen = torch.Generator()
    gen.manual_seed(cfg.random_state)
    full_ds = TensorDataset(X_train_t, y_train_t)
    train_inner_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_tr_inner, n_val], generator=gen
    )

    val_indices = list(val_ds.indices)
    train_indices = list(train_inner_ds.indices)

    X_val_t = X_train_t[val_indices]
    y_val_np = split.y_train[val_indices]
    y_test_np = split.y_test
    train_inner_X = split.X_train_scaled[train_indices]
    train_inner_y = split.y_train[train_indices]

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_inner_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size_eval, shuffle=False, num_workers=0, pin_memory=pin
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    return Loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        X_val_t=X_val_t,
        y_val_np=y_val_np,
        X_test_t=X_test_t,
        y_test_np=y_test_np,
        train_inner_X=train_inner_X,
        train_inner_y=train_inner_y,
    )
