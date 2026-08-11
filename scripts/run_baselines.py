#!/usr/bin/env python3
"""Train and evaluate the four classical baselines on California Housing.

Usage
-----
    python scripts/run_baselines.py
    python scripts/run_baselines.py --synthetic   # offline / no network
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ft_transformer.baselines import format_results_table, run_classical_baselines
from ft_transformer.data import load_raw_housing_data, make_synthetic_housing_data, prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of fetching California Housing.",
    )
    parser.add_argument(
        "--out",
        default="results/tables/baseline_results.json",
        help="Where to write the JSON results.",
    )
    args = parser.parse_args()

    if args.synthetic:
        X, y, _ = make_synthetic_housing_data()
    else:
        X, y, _ = load_raw_housing_data()

    split = prepare_data(X, y)
    results = run_classical_baselines(
        split.X_train_scaled, split.y_train, split.X_test_scaled, split.y_test
    )

    print(format_results_table(results))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
