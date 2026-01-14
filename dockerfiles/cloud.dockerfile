FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install some essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY configs/ configs/

WORKDIR /

# Install dependencies
RUN uv sync --all-groups --no-cache-dir

RUN mkdir -p models reports/figures

ENTRYPOINT ["uv", "run", "src/ml_ops/train.py", "+data.gcp=true"]
