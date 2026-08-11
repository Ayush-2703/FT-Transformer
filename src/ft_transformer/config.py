"""Typed configuration objects for the FT-Transformer pipeline.

Centralising every hyperparameter in dataclasses (rather than scattering literals
throughout the codebase) keeps `scripts/*.py`, `tests/*.py`, and `configs/*.yaml`
in sync and makes runs trivially reproducible from a single YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters for :class:`ft_transformer.model.FTTransformer`."""

    n_features: int = 8
    d_token: int = 64
    n_blocks: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    ffn_factor: float = 1.333

    def __post_init__(self) -> None:
        if self.d_token % self.n_heads != 0:
            raise ValueError(
                f"d_token ({self.d_token}) must be divisible by n_heads "
                f"({self.n_heads}) so every attention head sees an integer slice."
            )


@dataclass
class TrainConfig:
    """Optimisation hyperparameters for :func:`ft_transformer.engine.run_training`."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 150
    patience: int = 15
    seed: int = 42
    batch_size_train: int = 256
    batch_size_eval: int = 512
    grad_clip_norm: float = 1.0


@dataclass
class DataConfig:
    """Dataset split / preprocessing hyperparameters."""

    test_size: float = 0.2
    inner_val_size: float = 0.1
    random_state: int = 42


@dataclass
class ExperimentConfig:
    """Top-level config bundling model + training + data settings for one run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    name: str = "default"
