# Contributing

Thanks for considering a contribution! This started as an internship research project, so the
bar is: keep it reproducible, keep it honest, and keep the tests green.

## Setup

```bash
git clone https://github.com/Ayush-2703/ft-transformer-california-housing.git
cd ft-transformer-california-housing
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
make format   # auto-fix style
make lint     # ruff + black --check
make test     # offline test suite must pass
```

CI re-runs all of this (plus a Python 3.10/3.11/3.12 matrix and a real-dataset integration job),
so a green `make lint test` locally is a strong signal your PR will pass.

## Guidelines

- **New functionality needs tests.** Every module under `src/ft_transformer/` has a matching
  `tests/test_*.py`; keep it that way. Prefer `make_synthetic_housing_data()` over the real
  dataset in unit tests so the suite stays fast and network-independent — mark anything that
  genuinely needs the real data with `@pytest.mark.network`.
- **Report what actually happened.** If a change makes results worse, or an ablation doesn't show
  what you expected, document that rather than smoothing it over — this repo already has one
  documented paper/code discrepancy (see the README) and would rather have more honest notes
  than fewer.
- **Keep configs and code in sync.** If you change a default hyperparameter, update the matching
  entry in `configs/*.yaml` and, if it affects the headline numbers, the README results tables.
- **Small, focused PRs.** Architecture changes, new baselines, and tooling changes are easier to
  review separately.

## Reporting issues

Open a GitHub issue with:
- What you ran (exact command / config)
- What you expected vs. what happened
- Python version and `pip freeze | grep -E "torch|optuna|scikit-learn"` output

## Code of conduct

Be respectful and constructive. This is a small research repo, not a battleground.
