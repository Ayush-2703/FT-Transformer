# Reproducible CPU environment for the FT-Transformer pipeline.
# Build:  docker build -t ft-transformer -f docker/Dockerfile .
# Run:    docker run --rm -v "$(pwd)/results:/app/results" ft-transformer \
#             python scripts/run_full_pipeline.py --synthetic --fast

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies required by matplotlib's Agg backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY scripts ./scripts

RUN pip install -e .

COPY tests ./tests
COPY configs ./configs

RUN mkdir -p results/figures results/tables model_checkpoints

ENTRYPOINT ["python"]
CMD ["scripts/run_full_pipeline.py", "--synthetic", "--fast"]
