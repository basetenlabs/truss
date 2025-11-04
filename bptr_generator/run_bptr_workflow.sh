#!/bin/bash
set -e

echo "=== Generating bptr-manifest ==="
if [ -f "/app/model_cache_config.json" ]; then
    echo "Using model_cache config from ConfigMap"
    python /usr/local/bin/bptr_manifest_generator.py --model-cache /app/model_cache_config.json --output /static-bptr/static-bptr-manifest.json
else
    echo "Using full truss config"
    python /usr/local/bin/bptr_manifest_generator.py --config /app/config.yaml --output /static-bptr/static-bptr-manifest.json
fi

echo "=== Running truss-transfer-cli ==="
truss-transfer-cli
