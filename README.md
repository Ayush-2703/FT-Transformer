<div align="center">

![FT-Transformer](https://capsule-render.vercel.app/api?type=waving&color=0:1E1B4B,100:F97316&height=250&section=header&text=FT-Transformer&fontSize=60&fontColor=ffffff&fontAlignY=36&animation=fadeIn&desc=Feature%20Tokenizer%20%2B%20Transformer%20for%20Advance%20Tabular%20Regression&descSize=22&descAlignY=58)

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tests-50%20passing-10B981?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Coverage-98%25-10B981?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge"/>
</p>

<p>
  <a href="https://github.com/Ayush-2703/ft-transformer-california-housing/actions/workflows/ci.yml"><img src="https://github.com/Ayush-2703/ft-transformer-california-housing/actions/workflows/ci.yml/badge.svg"/></a>
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square"/>
  <img src="https://img.shields.io/badge/lint-ruff-46A6E0.svg?style=flat-square"/>
</p>

<p>
  <img src="https://img.shields.io/github/stars/Ayush-2703/ft-transformer-california-housing?style=social"/>
  <img src="https://img.shields.io/github/forks/Ayush-2703/ft-transformer-california-housing?style=social"/>
  <img src="https://img.shields.io/github/watchers/Ayush-2703/ft-transformer-california-housing?style=social"/>
</p>

<br/>

**A from-scratch PyTorch implementation of Feature Tokenizer + Transformer — with the discipline of a tested, CI-checked library, not just a research notebook.**

<br/>

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Ayush-2703/ft-transformer-california-housing/blob/main/notebooks/FT_Transformer_California_Housing.ipynb)
[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ayush-2703/ft-transformer-california-housing)

<br/>

**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**

</div>

---

## 📌 Table of Contents

- [Why This Repository](#-why-this-repository)
- [Results](#-results)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Development](#-development)
- [CI/CD](#-cicd)
- [Limitations & Future Work](#-limitations--future-work)
- [Citing](#-citing)
- [Author](#-author)

---

## 🎯 Why This Repository

Most "transformer for tabular data" repos stop at a notebook that ran once, on one seed, with no way to check whether the paper's claims survived translation into code. This one doesn't.

| What you'll find here | What you won't find here |
|---|---|
| ✅ A real 3-seed stability check, not a single lucky run | ❌ Cherry-picked results |
| ✅ Every claim in the internship report cross-checked against the code | ❌ Report and code silently disagreeing |
| ✅ 50 tests, 98% coverage, offline-by-default CI | ❌ "Trust me, it works" |
| ✅ Ablations on heads and depth, not just a final number | ❌ Hyperparameters pulled from thin air |
| ✅ `--synthetic` mode so anyone can verify the code with zero network access | ❌ A pipeline that only runs on the author's machine |
| ✅ Documented discrepancies, explained and resolved in favor of the code | ❌ Quietly patched-over inconsistencies |

> **TL;DR** — Feature-level tokenization + self-attention beats tuned Decision Trees, Ridge, and plain Linear Regression on this benchmark by **24.6% RMSE** (3-seed average), and the improvement holds up under ablation: more heads and more depth both help, with diminishing returns past 4 heads / 2 blocks.

Built as an internship research project, this repository packages that research notebook (`FT_Transformer_California_Housing.ipynb`) into a tested, documented, CI-checked Python library — implementing the architecture from [Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959) end to end: tokenizer, transformer blocks, training engine, baselines, ablations, hyperparameter search, and figure generation.

---

## 📊 Results

All numbers below are the actual measured outputs from the reference experiment (Google Colab, T4 GPU, `torch==2.x`, seeds fixed at 42 throughout). Reproduce with `make pipeline`, or step through `notebooks/FT_Transformer_California_Housing.ipynb`.

### Classical baselines vs. FT-Transformer

| Model                         | Train RMSE | Test RMSE  | Train R² | Test R² |
|--------------------------------|:---------:|:----------:|:--------:|:-------:|
| Linear Regression               | 0.7197    | 0.7456     | 0.6126   | 0.5758  |
| Ridge Regression                 | 0.7197    | 0.7456     | 0.6126   | 0.5758  |
| Decision Tree (default)           | 0.0000    | 0.7028     | 1.0000   | 0.6230  |
| Decision Tree (tuned)              | 0.4804    | 0.6454     | 0.8274   | 0.6822  |
| **FT-Transformer (default)**        | **0.4507** | **0.5043** | **0.8471** | **0.8060** |

*Default config: `d_token=64, n_blocks=2, n_heads=4, dropout=0.1` — 57,131 trainable parameters, early-stopped at epoch 92, best checkpoint from epoch 77, ~63s to train on a T4 GPU.*

<p align="center">
  <img src="results/figures/reference_run/figure1_rmse_comparison.png" alt="Train vs Test RMSE comparison" width="620">
</p>

### Ablation studies

<table>
<tr>
<td valign="top">

**Attention heads** (`d_token=64, n_blocks=2` fixed)

| Heads | Test RMSE  | Test R² |
|:-----:|:----------:|:-------:|
| 1     | 0.5139     | 0.7985  |
| **4** | **0.5043** | **0.8060** |
| 8     | 0.5184     | 0.7949  |

</td>
<td valign="top">

**Transformer depth** (`d_token=64, n_heads=4` fixed)

| Blocks | Test RMSE  | Test R² |
|:------:|:----------:|:-------:|
| 1      | 0.5381     | 0.7791  |
| **2**  | **0.5043** | **0.8060** |
| 3      | 0.5197     | 0.7939  |

</td>
</tr>
</table>

Both sweeps show the same pattern: capacity helps, then plateaus / mildly regresses. 4 heads and 2 blocks is the sweet spot at this dataset size — not "bigger is always better."

<p align="center">
  <img src="results/figures/reference_run/figure3_ablation_heatmap.png" alt="Ablation heatmap: test RMSE by heads x depth" width="560">
</p>

### Optuna hyperparameter search

50-trial TPE search over `d_token, n_blocks, n_heads, dropout, lr, weight_decay`, pruning any trial where `d_token % n_heads != 0`. Completed in ~2.8 hours on a T4.

| Hyperparameter | Best value |
|------------------|:---------:|
| `d_token`         | 64        |
| `n_blocks`        | 2         |
| `n_heads`         | 1         |
| `dropout`         | 0.0358    |
| `lr`              | 0.00675   |
| `weight_decay`    | ≈0.0001   |
| **Best val RMSE**  | **0.5038** |

Retraining that configuration across 3 seeds (0, 42, 123) to check stability:

| Seed | Train RMSE | Test RMSE | Test R² |
|:----:|:----------:|:---------:|:-------:|
| 0    | 0.4088     | 0.4770    | 0.8264  |
| 42   | 0.4000     | 0.4687    | 0.8324  |
| 123  | 0.4818     | 0.5150    | 0.7976  |
| **mean ± std** | — | **0.4869 ± 0.0202** | **0.8188 ± 0.0152** |

A low cross-seed standard deviation (±0.02 RMSE) indicates the result is a real, repeatable effect rather than a lucky seed.

<p align="center">
  <img src="results/figures/reference_run/figure2_loss_curves.png" alt="Training and validation loss curves" width="620">
</p>

> [!NOTE]
> **Parameter count — paper/code discrepancy.** The accompanying internship report (`docs/paper/`) lists 184,321 parameters for the default configuration. The notebook's own printed output, and `tests/test_model.py::test_default_config_parameter_count_matches_reference` in this repo, both confirm the correct figure is **57,131** for `d_token=64, n_blocks=2, n_heads=4`. The code is treated as ground truth here — this README and the config files use the verified number.
>
> **Permutation-invariance claim.** The report states that shuffling input feature order gives identical predictions, "eliminating the need for positional encoding." That's true of the attention mechanism's token-sequence handling, but not of the model as a whole: the tokenizer learns an *independent* weight/bias pair per feature index, so shuffling raw input columns **does** change predictions — see `test_cls_output_invariant_to_token_sequence_order` (the real, verified property) vs. `test_raw_feature_column_permutation_changes_prediction` (the documented counter-case) in `tests/test_model.py`.

---

## 🏗 Architecture

```
Input (B, 8)
      │
      ▼
NumericalTokenizer         token_i = W_i · x_i + b_i     (independent W_i, b_i per feature)
      │  (B, 8, d_token)
      ▼
Prepend learnable [CLS]    (B, 9, d_token)
      │
      ▼
N × TransformerBlock (Pre-LayerNorm)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ x' = x + MHA(LN₁(x))                                                    │
   │ x  = x' + FFN(LN₂(x'))                                                  │
   │   FFN: Linear(d→d_ff) → GELU → Linear(d_ff→d),  d_ff = floor(d × 4/3)   │
   └─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
LayerNorm(CLS) → Linear(1)
      │
      ▼
  ŷ  (B,)  — predicted median house value
```

**Design decisions**

| Choice | Why |
|---|---|
| No positional encoding | Feature order is not sequential, so none is used. |
| Pre-LayerNorm (Wang et al., 2019) | Avoids the early-training loss spikes a Post-Norm setup showed in preliminary runs. |
| AdamW + cosine-annealing LR | Decoupled weight decay, smooth LR decay to convergence. |
| Gradient-norm clipping (1.0) | Stabilizes early training against attention-score outliers. |
| Early stopping (patience=15) | Best checkpoint restored on validation RMSE, not train loss. |

Full architectural derivation, design-decision rationale, and literature review are in [`docs/paper/`](docs/paper/).

---

## 📁 Repository Structure

<details open>
<summary><b>Show project structure</b></summary>

```
.
├── src/ft_transformer/       # the installable package
│   ├── tokenizer.py           # NumericalTokenizer
│   ├── blocks.py               # Pre-Norm TransformerBlock
│   ├── model.py                 # full FTTransformer
│   ├── data.py                   # loading, leakage-free scaling, DataLoaders
│   ├── baselines.py               # 4 classical sklearn baselines
│   ├── engine.py                    # train/eval loop, run_training orchestrator
│   ├── ablation.py                   # attention-head / depth sweeps
│   ├── hpo.py                         # Optuna TPE search
│   ├── visualize.py                    # the 4 report figures
│   ├── config.py                        # ModelConfig / TrainConfig / DataConfig
│   └── utils.py                          # seeding, device selection
├── scripts/                  # thin CLIs over the package (see below)
├── tests/                    # 50 tests, 98% coverage, offline by default
├── configs/                  # default.yaml, optuna_best.yaml
├── notebooks/                # the original research notebook
├── docs/paper/                # the research report (PDF + DOCX)
├── docker/Dockerfile
├── .github/workflows/ci.yml  # lint + multi-version test + smoke + network jobs
└── results/                  # figures/tables (reference run figures checked in)
```

</details>

---

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ayush-2703/ft-transformer-california-housing.git
cd ft-transformer-california-housing
```

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Run it

```bash
# Fast, fully offline correctness check (synthetic data, ~30s):
make smoke

# Full reproduction against the real California Housing dataset (needs network,
# ~20 min default run / ~3 hrs including the 50-trial Optuna search on CPU):
make pipeline
```

Or run each stage independently:

```bash
python scripts/run_baselines.py                                 # 4 classical baselines
python scripts/train.py --d-token 64 --n-blocks 2 --n-heads 4    # default FT-Transformer
python scripts/run_ablation.py                                   # heads + depth sweeps
python scripts/run_hpo.py --n-trials 50                          # Optuna search + 3-seed eval
python scripts/run_full_pipeline.py                              # everything, plus all 4 figures
```

Every script accepts `--synthetic` to run against generated data with the same shape (no network access required) — useful for offline development or a quick sanity check.

### 4. Or skip setup entirely — open in Colab

Click the Colab badge at the top of this README, or go directly to:

```
https://colab.research.google.com/github/Ayush-2703/ft-transformer-california-housing
```

### 5. Docker

```bash
docker build -t ft-transformer -f docker/Dockerfile .
docker run --rm -v "$(pwd)/results:/app/results" ft-transformer \
    python scripts/run_full_pipeline.py --synthetic --fast
```

---

## 🛠 Development

```bash
pip install -e ".[dev]"
pre-commit install        # auto-run lint/format on every commit

make lint                 # ruff + black --check
make format                # ruff --fix + black
make test                   # offline unit tests (pytest, ~20s)
make test-cov                # same, with a coverage report
```

The **non-network** test suite (`make test`) covers the tokenizer, transformer block, full model (including a genuine permutation-invariance property check), the training engine, classical baselines, the data pipeline, ablation and Optuna search logic, figure generation, and CLI smoke tests — all against synthetic data, so it runs identically offline and in CI in under 30 seconds. A separate, explicitly-marked `pytest -m network` job fetches the real dataset to check the download path still works.

---

## 🔁 CI/CD

Every push and PR to `main` runs four jobs (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

| Job                 | What it checks |
|----------------------|----------------|
| `lint`                | `ruff check` + `black --check` |
| `test`                 | Full offline test suite × Python 3.10 / 3.11 / 3.12, with coverage |
| `cli-smoke`             | Runs `scripts/run_full_pipeline.py --synthetic --fast` end-to-end and uploads the produced figures/tables as an artifact |
| `data-integration`       | Fetches the *real* California Housing dataset and re-validates its shape (`pytest -m network`) |

---

## 🧭 Limitations & Future Work

Carried over honestly from the original report, since they still apply to this repo:

- Only compared against Linear/Ridge/Decision-Tree baselines — no XGBoost/LightGBM/CatBoost yet.
- Single dataset (California Housing) — generalization to other tabular tasks is untested.
- No explainability tooling (SHAP, attention-weight visualization) yet.
- The Optuna search space and trial budget (50 trials, CPU-hour constrained originally) is almost certainly not the global optimum.

Contributions on any of the above are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## 📜 Citing

If this implementation is useful for your work, please cite both the original FT-Transformer paper and this repository — see [`CITATION.cff`](CITATION.cff).

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.

---

## 👤 Author

<div align="center">

### Ayush Kumar Singh

*Researcher in Adversarial ML, Geospatial AI, and LLM/NLP Systems*

[![GitHub](https://img.shields.io/badge/GitHub-Ayush%20Kumar%20Singh-181717?style=for-the-badge&logo=github)](https://github.com/Ayush-2703)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush%20Kumar%20Singh-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/ayushsingh2703)
[![Email](https://img.shields.io/badge/Email-Ayush%20Kumar%20Singh-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ab49ayush@gmail.com)

</div>

---

<div align="center">

**If this repository helped you, please consider giving it a ⭐**
*It takes 2 seconds and helps others discover it.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1E1B4B,100:F97316&height=100&section=footer" width="100%"/>

</div>
