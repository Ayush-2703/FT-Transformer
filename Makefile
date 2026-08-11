.PHONY: help install install-dev lint format test test-cov smoke baselines train ablation hpo pipeline pipeline-fast clean

help:
	@echo "Available targets:"
	@echo "  install       Install the package (runtime deps only)"
	@echo "  install-dev   Install the package with dev/test/lint deps"
	@echo "  lint          Run ruff + black --check"
	@echo "  format        Auto-format with black and ruff --fix"
	@echo "  test          Run the offline unit test suite"
	@echo "  test-cov      Run tests with a coverage report"
	@echo "  smoke         Fast end-to-end smoke test on synthetic data"
	@echo "  baselines     Train the 4 classical baselines on real data"
	@echo "  train         Train the default FT-Transformer on real data"
	@echo "  ablation      Run the attention-head / depth ablation studies"
	@echo "  hpo           Run the 50-trial Optuna search + 3-seed final eval"
	@echo "  pipeline      Run the full reproduction pipeline (slow, real data)"
	@echo "  pipeline-fast Run the full pipeline on synthetic data (CI-friendly)"
	@echo "  clean         Remove caches, build artifacts, and generated results"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src scripts tests
	black --check src scripts tests

format:
	ruff check --fix src scripts tests
	black src scripts tests

test:
	pytest -v

test-cov:
	pytest --cov=ft_transformer --cov-report=term-missing --cov-report=html -v

smoke:
	python scripts/run_full_pipeline.py --synthetic --fast

baselines:
	python scripts/run_baselines.py

train:
	python scripts/train.py

ablation:
	python scripts/run_ablation.py

hpo:
	python scripts/run_hpo.py

pipeline:
	python scripts/run_full_pipeline.py

pipeline-fast:
	python scripts/run_full_pipeline.py --synthetic --fast

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov build dist *.egg-info
	rm -f results/tables/*.json results/figures/*.png model_checkpoints/*.pt
