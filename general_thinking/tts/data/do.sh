#!/bin/bash
set -euo pipefail


# run_server.sh invokes `python` directly (e.g. to snapshot_download the
# ForcedAligner). The Truss image only ships `python3`, so symlink it.
if ! command -v python >/dev/null 2>&1; then
    ln -sf "$(command -v python3)" /usr/local/bin/python
fi

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 /app/vllm-omni/examples/online_serving/qwen3_tts/run_server.sh --timestamps --profile ttfa_32 "$@"