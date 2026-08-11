"""Classical regression baselines used as the comparison point for FT-Transformer.

Trains and evaluates four scikit-learn regressors on the same 80/20 split used
by the FT-Transformer: Linear Regression, Ridge (alpha=1), an untuned Decision
Tree, and a Decision Tree tuned via 5-fold GridSearchCV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

DEFAULT_DT_PARAM_GRID = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
}


def compute_metrics(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute train/test RMSE and R^2 for a fitted scikit-learn estimator."""
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    tr_rmse = float(np.sqrt(mean_squared_error(y_train, tr_pred)))
    te_rmse = float(np.sqrt(mean_squared_error(y_test, te_pred)))
    tr_r2 = float(r2_score(y_train, tr_pred))
    te_r2 = float(r2_score(y_test, te_pred))
    return tr_rmse, te_rmse, tr_r2, te_r2


def run_classical_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dt_param_grid: dict | None = None,
    random_state: int = 42,
    cv_folds: int = 5,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate all four classical baselines.

    Returns
    -------
    dict
        Keyed by model name (``"Linear Reg"``, ``"Ridge"``, ``"DT Default"``,
        ``"DT Tuned"``), each mapping to a dict with ``train_rmse``,
        ``test_rmse``, ``train_r2``, ``test_r2`` (and ``best_params`` for the
        tuned tree).
    """
    dt_param_grid = dt_param_grid or DEFAULT_DT_PARAM_GRID
    results: dict[str, dict[str, Any]] = {}

    # (a) Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    tr, te, tr2, te2 = compute_metrics(lr_model, X_train, y_train, X_test, y_test)
    results["Linear Reg"] = dict(train_rmse=tr, test_rmse=te, train_r2=tr2, test_r2=te2)

    # (b) Ridge
    rg_model = Ridge(alpha=1.0)
    rg_model.fit(X_train, y_train)
    tr, te, tr2, te2 = compute_metrics(rg_model, X_train, y_train, X_test, y_test)
    results["Ridge"] = dict(train_rmse=tr, test_rmse=te, train_r2=tr2, test_r2=te2)

    # (c) Decision Tree -- default, untuned
    dt_def = DecisionTreeRegressor(random_state=random_state)
    dt_def.fit(X_train, y_train)
    tr, te, tr2, te2 = compute_metrics(dt_def, X_train, y_train, X_test, y_test)
    results["DT Default"] = dict(train_rmse=tr, test_rmse=te, train_r2=tr2, test_r2=te2)

    # (d) Decision Tree -- GridSearchCV (k-fold, MSE)
    gs = GridSearchCV(
        DecisionTreeRegressor(random_state=random_state),
        param_grid=dt_param_grid,
        scoring="neg_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    tr, te, tr2, te2 = compute_metrics(gs.best_estimator_, X_train, y_train, X_test, y_test)
    results["DT Tuned"] = dict(
        train_rmse=tr, test_rmse=te, train_r2=tr2, test_r2=te2, best_params=gs.best_params_
    )

    return results


def format_results_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the baseline results dict as a fixed-width text table."""
    width = 72
    lines = ["=" * width]
    lines.append(
        f"{'Model':<18} {'Train RMSE':>11} {'Test RMSE':>10} {'Train R2':>9} {'Test R2':>9}"
    )
    lines.append("=" * width)
    for name, res in results.items():
        lines.append(
            f"{name:<18} {res['train_rmse']:>11.4f} {res['test_rmse']:>10.4f} "
            f"{res['train_r2']:>9.4f} {res['test_r2']:>9.4f}"
        )
    lines.append("=" * width)
    return "\n".join(lines)
