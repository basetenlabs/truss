#!/bin/bash
set -eux

# System deps (PyTorch image lacks curl/rsync/git; cuDNN dev headers needed for TE build)
apt-get update -qq && apt-get install -y --no-install-recommends \
    rsync curl ca-certificates git libcudnn9-dev-cuda-12

# Install uv (manages Python deps + venv)
command -v uv &>/dev/null || {
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

cd /workspace

# Shared cache across nodes and between job runs
export UV_CACHE_DIR="${BT_RW_CACHE_DIR}/uv_cache"
export UV_LINK_MODE=copy
mkdir -p "${UV_CACHE_DIR}"

# Install all Python deps into a clean venv (ignores system torch)
uv sync

# CUDA extensions need torch at build time - install after uv sync
uv pip install flash-attn --no-build-isolation || echo "WARN: flash-attn unavailable"
uv pip install "transformer-engine[pytorch]" --no-build-isolation || echo "WARN: transformer-engine unavailable"

# Activate venv for all subsequent commands
source .venv/bin/activate
echo "Python: $(python --version) at $(which python)"
echo "swift: $(which swift)"
