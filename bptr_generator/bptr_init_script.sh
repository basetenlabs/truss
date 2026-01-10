#!/bin/bash
set -e

BPTR_DIR="${BPTR_DIR:-/cache/model/bptr}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/cache/model/model_cache}"
APP_MODEL_CACHE="${APP_MODEL_CACHE:-/app/model_cache}"
MODEL_CACHE_CONFIG="${MODEL_CACHE_CONFIG:-/app/model_cache_config.json}"

mkdir -p "$BPTR_DIR" "$MODEL_CACHE_DIR"
ln -sf "$BPTR_DIR" /static-bptr
ln -sf "$BPTR_DIR" /bptr

python /usr/local/bin/bptr_manifest_generator.py \
  --model-cache "$MODEL_CACHE_CONFIG" \
  --output "$BPTR_DIR/static-bptr-manifest.json" \
  --model-path "$MODEL_CACHE_DIR"

ln -sf "$BPTR_DIR/static-bptr-manifest.json" "$BPTR_DIR/bptr-manifest.json"
ln -sf "$BPTR_DIR/static-bptr-manifest.json" "$BPTR_DIR/bptr-manifest"

if [ ! -f "$BPTR_DIR/static-bptr-manifest.json" ]; then
  echo "ERROR: Manifest not found" >&2
  ls -la "$BPTR_DIR" || true
  exit 1
fi

truss-transfer-cli

mkdir -p "$APP_MODEL_CACHE"
ln -sf "$MODEL_CACHE_DIR"/* "$APP_MODEL_CACHE"/ 2>/dev/null || true
