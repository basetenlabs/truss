#!/bin/bash
# Derives SLURM_DIR, the multinode rendezvous dir. Sourced by install_slurm.sh.
# Per-job dir: the project cache is shared, so concurrent jobs must not share it.
if [ -z "${BT_TRAINING_JOB_ID}" ]; then
    echo "ERROR: BT_TRAINING_JOB_ID must be set to scope the SLURM rendezvous dir" >&2
    exit 1
fi
SLURM_DIR="${BT_PROJECT_CACHE_DIR}/slurm_${BT_TRAINING_JOB_ID}"
