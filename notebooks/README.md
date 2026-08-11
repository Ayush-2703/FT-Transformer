# Notebooks

`FT_Transformer_California_Housing.ipynb` is the original research notebook this repository was
built from — the single source of truth for every number in the top-level README. It runs
top-to-bottom on a free Google Colab T4 GPU in roughly 3 hours (dominated by the 50-trial Optuna
search); CPU-only execution works but is proportionally slower.

For day-to-day development, prefer the packaged modules and CLI scripts under `src/ft_transformer/`
and `scripts/` — they're unit-tested, importable, and don't require re-running an entire notebook
to check a change. This notebook is kept as the canonical, linear reference reproduction.
