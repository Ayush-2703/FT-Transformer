# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-08-11

### Added
- Initial packaged release: `src/ft_transformer/` extracted and modularized from the original
  research notebook (`notebooks/FT_Transformer_California_Housing.ipynb`).
- Full test suite (50 tests, 98% line coverage on the non-network path), including a genuine
  permutation-invariance property check for the CLS-token readout.
- CLI scripts for baselines, single-config training, ablations, Optuna search, and the full
  pipeline, each with a `--synthetic` offline mode.
- GitHub Actions CI: lint, multi-version test matrix (3.10/3.11/3.12), fast synthetic-data smoke
  test, and a separate real-dataset network-integration job.
- `Dockerfile`, `Makefile`, `pre-commit` config, `pyproject.toml` packaging.
- `configs/default.yaml` and `configs/optuna_best.yaml` capturing the two headline configurations.

### Fixed
- Corrected a parameter-count discrepancy between the research report (184,321) and the model's
  actual trainable-parameter count (57,131, for `d_token=64, n_blocks=2, n_heads=4`) — verified by
  `tests/test_model.py::test_default_config_parameter_count_matches_reference` and pinned as the
  documented figure throughout the README and configs.
- Corrected an imprecise permutation-invariance claim in the report ("shuffling input features
  gives identical predictions"): the true, verified property is invariance to *token sequence
  order after tokenization*, not to raw input column order (the tokenizer intentionally learns
  per-feature-index parameters). Both the correct property and the documented counter-case are
  covered by tests.
