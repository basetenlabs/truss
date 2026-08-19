FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-dev curl git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY vendor/ vendor/
COPY src/ src/

RUN uv sync --extra worker --no-dev

ENTRYPOINT [".venv/bin/python", "-m", "trainers_server.dp_worker.main"]
