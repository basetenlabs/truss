#!/bin/bash
# Entry point for multi-node workstation SLURM setup.
# Dispatches to controller or worker based on BT_NODE_RANK.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common setup: install SLURM, munge, detect GPUs
source "${SCRIPT_DIR}/install_slurm.sh"

if [ "${BT_NODE_RANK}" = "0" ]; then
    source "${SCRIPT_DIR}/setup_controller.sh"
else
    source "${SCRIPT_DIR}/setup_worker.sh"
fi
