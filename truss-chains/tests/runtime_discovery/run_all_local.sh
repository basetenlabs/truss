#!/usr/bin/env bash
# Run every runtime_discovery example's local test. CPU-only.
set -euo pipefail
cd "$(dirname "$0")"
for dir in 0*; do
  for test_file in "$dir"/test_*.py; do
    if [[ -f "$test_file" ]]; then
      echo "=== $dir / $(basename "$test_file") ==="
      uv run pytest "$test_file" -v
    fi
  done
done
