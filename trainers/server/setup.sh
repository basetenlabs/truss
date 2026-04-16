#!/usr/bin/env bash
# setup.sh — install the trainers-server worker environment.
#
# Why not `uv sync`?
# ------------------
# uv resolves the entire workspace before installing. megatron-bridge's
# pyproject.toml declares `megatron-core = { path = "3rdparty/Megatron-LM/" }`
# in [tool.uv.sources], which conflicts with any top-level megatron-core source
# uv finds in the workspace.  We work around this by using `uv pip install`
# directly — it skips workspace resolution and installs packages as specified.
#
# Install order matters:
#   torch must be installed before megatron-bridge (bridge's build backend needs it).
#   megatron-core (from the bridge submodule) must be installed before megatron-bridge.
#
# Usage:
#   cd trainers/server
#   bash setup.sh          # installs worker + dev extras
#   bash setup.sh --no-dev # skips dev extras (pytest etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Submodules ─────────────────────────────────────────────────────────────────
# Pulls vendor/megatron-bridge and its nested 3rdparty/Megatron-LM submodule.
echo "==> Initialising git submodules..."
git submodule update --init --recursive

NO_DEV=false
for arg in "$@"; do
  [[ "$arg" == "--no-dev" ]] && NO_DEV=true
done

# ── System dependencies ────────────────────────────────────────────────────────
# transformer-engine[pytorch] builds a C extension and needs these.
echo "==> Checking system build dependencies..."
MISSING_PKGS=()
for pkg in python3-dev cmake ninja-build; do
  dpkg -s "$pkg" &>/dev/null || MISSING_PKGS+=("$pkg")
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  echo "    Installing: ${MISSING_PKGS[*]}"
  apt-get install -y "${MISSING_PKGS[@]}"
fi

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
  echo "==> Creating virtual environment..."
  uv venv
fi

UV="uv pip install --python .venv/bin/python"

# ── PyTorch (must come first — megatron-bridge build backend requires it) ──────
echo "==> Installing PyTorch (cu128)..."
$UV torch --index-url https://download.pytorch.org/whl/cu128

# ── megatron-core from the bridge's bundled submodule ─────────────────────────
# The bridge was written against 3rdparty/Megatron-LM and requires its exact API.
echo "==> Installing megatron-core from vendor/megatron-bridge/3rdparty/Megatron-LM..."
$UV -e vendor/megatron-bridge/3rdparty/Megatron-LM

# ── megatron-bridge (skip its own deps — we manage them here) ─────────────────
echo "==> Installing megatron-bridge..."
$UV -e vendor/megatron-bridge --no-deps

# ── Heavy ML deps ─────────────────────────────────────────────────────────────
echo "==> Installing transformer-engine, nvidia-modelopt, and supporting libs..."
$UV "transformer-engine[pytorch]" nvidia-modelopt omegaconf einops safetensors

# ── Main deps declared in pyproject.toml ──────────────────────────────────────
echo "==> Installing trainers-server runtime deps..."
$UV transformers fastapi "pydantic>=2" httpx uvicorn

# ── vLLM ──────────────────────────────────────────────────────────────────────
echo "==> Installing vLLM..."
$UV "vllm>=0.12.0"

# ── trainers-server itself ─────────────────────────────────────────────────────
echo "==> Installing trainers-server (editable)..."
$UV -e . --no-deps

# ── Dev extras (pytest) ───────────────────────────────────────────────────────
if [[ "$NO_DEV" == false ]]; then
  echo "==> Installing dev extras..."
  $UV pytest httpx
fi

echo ""
echo "Done. Activate with: source .venv/bin/activate"
echo "Run tests:           .venv/bin/python -m pytest tests/ -m 'not gpu'"
echo "Run GPU tests:       .venv/bin/python -m pytest tests/ -m gpu -v"
