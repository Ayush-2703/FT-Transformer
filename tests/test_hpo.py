"""Unit tests for :mod:`ft_transformer.hpo`."""

from __future__ import annotations

from ft_transformer.hpo import best_config_from_study, run_search


def test_run_search_completes_and_prunes_invalid_configs(synthetic_loaders, device):
    study = run_search(
        synthetic_loaders,
        n_features=8,
        device=device,
        n_trials=6,
        seed=42,
        max_epochs=2,
        patience=1,
        show_progress_bar=False,
    )
    assert len(study.trials) == 6
    assert study.best_value >= 0
    # Every completed trial's params must respect the divisibility constraint.
    for trial in study.trials:
        if trial.state.name == "COMPLETE":
            assert trial.params["d_token"] % trial.params["n_heads"] == 0


def test_best_config_from_study_builds_valid_configs(synthetic_loaders, device):
    study = run_search(
        synthetic_loaders,
        n_features=8,
        device=device,
        n_trials=4,
        seed=42,
        max_epochs=2,
        patience=1,
        show_progress_bar=False,
    )
    cfg = best_config_from_study(study, n_features=8)
    assert cfg["model"].n_features == 8
    assert cfg["model"].d_token % cfg["model"].n_heads == 0
    assert "lr" in cfg["train_partial"] and "weight_decay" in cfg["train_partial"]
