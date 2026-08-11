"""Unit tests for :mod:`ft_transformer.baselines`."""

from __future__ import annotations

from ft_transformer.baselines import format_results_table, run_classical_baselines


def test_run_classical_baselines_returns_all_four_models(synthetic_split):
    results = run_classical_baselines(
        synthetic_split.X_train_scaled,
        synthetic_split.y_train,
        synthetic_split.X_test_scaled,
        synthetic_split.y_test,
        cv_folds=2,
    )
    assert set(results.keys()) == {"Linear Reg", "Ridge", "DT Default", "DT Tuned"}


def test_baseline_metrics_are_sane(synthetic_split):
    results = run_classical_baselines(
        synthetic_split.X_train_scaled,
        synthetic_split.y_train,
        synthetic_split.X_test_scaled,
        synthetic_split.y_test,
        cv_folds=2,
    )
    for name, res in results.items():
        assert res["train_rmse"] >= 0, name
        assert res["test_rmse"] >= 0, name
        assert res["train_r2"] <= 1.0, name
        assert res["test_r2"] <= 1.0, name


def test_tuned_tree_reports_best_params(synthetic_split):
    results = run_classical_baselines(
        synthetic_split.X_train_scaled,
        synthetic_split.y_train,
        synthetic_split.X_test_scaled,
        synthetic_split.y_test,
        cv_folds=2,
    )
    assert "best_params" in results["DT Tuned"]
    assert "max_depth" in results["DT Tuned"]["best_params"]


def test_format_results_table_contains_every_model(synthetic_split):
    results = run_classical_baselines(
        synthetic_split.X_train_scaled,
        synthetic_split.y_train,
        synthetic_split.X_test_scaled,
        synthetic_split.y_test,
        cv_folds=2,
    )
    table = format_results_table(results)
    for name in results:
        assert name in table
