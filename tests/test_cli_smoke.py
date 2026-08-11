"""End-to-end smoke tests that invoke the CLI scripts as subprocesses.

These exercise the full argparse -> pipeline -> file-output path without
touching the network (everything uses ``--synthetic``), catching integration
bugs that pure unit tests miss (e.g. a script importing a function that no
longer exists, or writing to the wrong path).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_run_baselines_script(tmp_path):
    out_file = tmp_path / "baseline_results.json"
    result = _run("run_baselines.py", "--synthetic", "--out", str(out_file))
    assert result.returncode == 0, result.stderr
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert set(data.keys()) == {"Linear Reg", "Ridge", "DT Default", "DT Tuned"}


def test_train_script(tmp_path):
    ckpt = tmp_path / "model.pt"
    out_file = tmp_path / "train_result.json"
    result = _run(
        "train.py",
        "--synthetic",
        "--max-epochs",
        "2",
        "--patience",
        "1",
        "--d-token",
        "16",
        "--n-blocks",
        "1",
        "--n-heads",
        "2",
        "--checkpoint",
        str(ckpt),
        "--out",
        str(out_file),
    )
    assert result.returncode == 0, result.stderr
    assert ckpt.exists()
    data = json.loads(out_file.read_text())
    assert "test_rmse" in data and data["test_rmse"] >= 0


def test_run_ablation_script(tmp_path):
    out_file = tmp_path / "ablation_results.json"
    result = _run(
        "run_ablation.py",
        "--synthetic",
        "--max-epochs",
        "2",
        "--patience",
        "1",
        "--heads-grid",
        "1",
        "2",
        "--blocks-grid",
        "1",
        "2",
        "--out",
        str(out_file),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out_file.read_text())
    assert "heads_ablation" in data and "depth_ablation" in data


def test_run_hpo_script(tmp_path):
    ckpt = tmp_path / "model_optuna_best.pt"
    out_file = tmp_path / "hpo_results.json"
    result = _run(
        "run_hpo.py",
        "--synthetic",
        "--n-trials",
        "2",
        "--max-epochs",
        "2",
        "--patience",
        "1",
        "--final-max-epochs",
        "2",
        "--final-patience",
        "1",
        "--seeds",
        "0",
        "1",
        "--checkpoint",
        str(ckpt),
        "--out",
        str(out_file),
    )
    assert result.returncode == 0, result.stderr
    assert ckpt.exists()
    data = json.loads(out_file.read_text())
    assert "best_params" in data
