#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_HARNESS_DIR="${BT_PROJECT_CACHE_DIR}/slurm_harness"

# Clean up all stale state from previous runs.
rm -rf "$SLURM_HARNESS_DIR/jobs"
rm -f "$SLURM_HARNESS_DIR"/controller_* \
      "$SLURM_HARNESS_DIR"/slurm.conf \
      "$SLURM_HARNESS_DIR"/munge.key \
      "$SLURM_HARNESS_DIR"/munge_test_cred

# Install SLURM and munge
source "${SCRIPT_DIR}/../shared/install_slurm.sh"

# Configure truss remote so workers can be pushed from the login node
if [ -n "${BASETEN_API_KEY:-}" ]; then
    python3 -c "import truss, os; truss.login(os.environ['BASETEN_API_KEY'])"
    echo "Truss remote configured for login node."
fi

# Generate munge key and share it via project cache (atomic write)
dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key 2>/dev/null
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key
cp /etc/munge/munge.key "$SLURM_HARNESS_DIR/munge.key.tmp"
mv "$SLURM_HARNESS_DIR/munge.key.tmp" "$SLURM_HARNESS_DIR/munge.key"

# Start munge
pkill -9 munged 2>/dev/null || true
sleep 1
service munge start || munged --force

# Get controller IP
CONTROLLER_IP=$(hostname -I | awk '{print $1}')
CONTROLLER_HOSTNAME=$(hostname -s)

echo "$CONTROLLER_IP" > "$SLURM_HARNESS_DIR/controller_ip"
echo "$CONTROLLER_HOSTNAME" > "$SLURM_HARNESS_DIR/controller_hostname"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Create required directories
mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
chmod 755 /var/spool/slurmctld /var/spool/slurmd

# Baseten hostname convention:
#   baseten-training-job-<job_id>-multinode-0       (rank 0)
#   baseten-training-job-<job_id>-multinode-0-<N>   (rank N > 0)
worker_hostname_for_rank() {
    local job_id="$1"
    local rank="$2"
    if [ "$rank" -eq 0 ]; then
        echo "baseten-training-job-${job_id}-multinode-0"
    else
        echo "baseten-training-job-${job_id}-multinode-0-${rank}"
    fi
}

# Write the base slurm.conf header (no nodes yet).
# Node and partition lines are appended by add_job_nodes().
write_slurm_conf_header() {
    cat > /etc/slurm/slurm.conf <<SLURMCONF
ClusterName=baseten-slurm
SlurmctldHost=${CONTROLLER_HOSTNAME}(${CONTROLLER_IP})

SlurmUser=root

MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
StateSaveLocation=/var/spool/slurmctld

SchedulerType=sched/builtin
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

SlurmctldDebug=info
SlurmdDebug=info
SlurmdParameters=config_overrides

SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
SlurmctldTimeout=120
WaitTime=0

GresTypes=gpu

PartitionName=DEFAULT Default=YES MaxTime=INFINITE State=UP
SLURMCONF
}

# Append nodes from a single job to slurm.conf and update the partition line.
# Uses ALL_PARTITION_NODES (global accumulator) to track every node across jobs.
add_job_nodes() {
    local job_id="$1"
    local job_dir="$SLURM_HARNESS_DIR/jobs/$job_id"
    local node_count
    node_count=$(cat "$job_dir/node_count")

    local new_nodes=""
    for i in $(seq 0 $((node_count - 1))); do
        local worker_ip worker_host worker_cpus worker_gpus
        worker_ip=$(cat "$job_dir/worker_${i}_ip")
        worker_host=$(worker_hostname_for_rank "$job_id" "$i")

        # Update /etc/hosts (remove stale entry first)
        sed -i "/ ${worker_host}$/d" /etc/hosts 2>/dev/null || true
        echo "${worker_ip} ${worker_host}" >> /etc/hosts

        # Read actual CPU count from worker's report, or default to 224 (H200)
        worker_cpus=224
        if [ -f "$job_dir/worker_${i}_cpus" ]; then
            worker_cpus=$(cat "$job_dir/worker_${i}_cpus")
        fi
        # Read actual GPU count from worker, or fall back to GPUS_PER_NODE
        worker_gpus="${GPUS_PER_NODE}"
        if [ -f "$job_dir/worker_${i}_gpus" ]; then
            worker_gpus=$(cat "$job_dir/worker_${i}_gpus")
        fi

        # Insert NodeName line before the PartitionName=gpu line (or append if
        # this is the first job and no partition line exists yet).
        if grep -q "^PartitionName=gpu " /etc/slurm/slurm.conf 2>/dev/null; then
            sed -i "/^PartitionName=gpu /i NodeName=${worker_host} NodeAddr=${worker_ip} CPUs=${worker_cpus} RealMemory=100000 Gres=gpu:${worker_gpus} State=UNKNOWN" /etc/slurm/slurm.conf
        else
            echo "NodeName=${worker_host} NodeAddr=${worker_ip} CPUs=${worker_cpus} RealMemory=100000 Gres=gpu:${worker_gpus} State=UNKNOWN" >> /etc/slurm/slurm.conf
        fi

        if [ -z "$new_nodes" ]; then
            new_nodes="$worker_host"
        else
            new_nodes="${new_nodes},${worker_host}"
        fi
    done

    # Update the global node accumulator and rewrite the partition line
    if [ -z "$ALL_PARTITION_NODES" ]; then
        ALL_PARTITION_NODES="$new_nodes"
    else
        ALL_PARTITION_NODES="${ALL_PARTITION_NODES},${new_nodes}"
    fi

    # Replace or append the partition line
    if grep -q "^PartitionName=gpu " /etc/slurm/slurm.conf 2>/dev/null; then
        sed -i "s|^PartitionName=gpu .*|PartitionName=gpu Nodes=${ALL_PARTITION_NODES} Default=YES MaxTime=INFINITE State=UP|" /etc/slurm/slurm.conf
    else
        echo "PartitionName=gpu Nodes=${ALL_PARTITION_NODES} Default=YES MaxTime=INFINITE State=UP" >> /etc/slurm/slurm.conf
    fi

    cp /etc/slurm/slurm.conf "$SLURM_HARNESS_DIR/slurm.conf"
    echo "Added ${node_count} node(s) from job ${job_id}. Total partition nodes: ${ALL_PARTITION_NODES}"
}

# Start slurmctld for the first time. Clears stale state from prior login runs.
# Called ONCE — subsequent jobs use scontrol reconfigure instead.
start_slurmctld() {
    rm -f /var/spool/slurmctld/node_state /var/spool/slurmctld/node_state.old
    rm -f /var/spool/slurmctld/job_state /var/spool/slurmctld/job_state.old

    slurmctld -D &
    SLURMCTLD_PID=$!
    sleep 3

    if kill -0 "$SLURMCTLD_PID" 2>/dev/null; then
        echo "slurmctld running (PID ${SLURMCTLD_PID})"
        return 0
    else
        echo "ERROR: slurmctld failed to start"
        cat /var/log/slurm/slurmctld.log || true
        return 1
    fi
}

# Check whether a job directory has all its workers registered.
job_workers_ready() {
    local job_dir="$1"
    [ -f "$job_dir/node_count" ] || return 1
    local node_count
    node_count=$(cat "$job_dir/node_count")
    for i in $(seq 0 $((node_count - 1))); do
        [ -f "$job_dir/worker_${i}_ip" ] && [ -f "$job_dir/worker_${i}_gpus" ] || return 1
    done
    return 0
}

# Self-test: push a worker job from this login node before waiting for workers.
# Reads config from runtime_config.json (already on disk from the push).
if [ -f /workspace/runtime_config.json ]; then
    eval "$(python3 -c "
import json
c = json.load(open('/workspace/runtime_config.json'))
print(f'SELF_TEST={c.get(\"self_test\", False)}')
print(f'PROJECT={c.get(\"project_name\", \"slurm-harness\")}')
print(f'PARTITION={c.get(\"partition\", \"H200\")}')
print(f'SELF_TEST_WORKERS={c.get(\"node_count\", 1)}')
" 2>/dev/null)" || SELF_TEST=False
    if [ "$SELF_TEST" = "True" ]; then
        echo "SELF_TEST: Pushing a worker job from login node..."
        truss train slurm sbatch --wrap "echo SELF_TEST_OK && hostname && nvidia-smi -L && sleep 30 && echo SELF_TEST_DONE" \
            --project "$PROJECT" -p "$PARTITION" --gres "gpu:${GPUS_PER_NODE}" -N "${SELF_TEST_WORKERS}" 2>&1 \
            || echo "SELF_TEST: sbatch failed with exit code $?"
        echo "SELF_TEST: Worker push initiated, now waiting for it to register..."
    fi
fi

# --- Wait for the first job to appear and register ---
echo "Waiting for first job directory in ${SLURM_HARNESS_DIR}/jobs/..."
MAX_WAIT=600
WAITED=0
FIRST_JOB_ID=""
while true; do
    for job_dir in "$SLURM_HARNESS_DIR/jobs"/*/; do
        [ -d "$job_dir" ] || continue
        FIRST_JOB_ID=$(basename "$job_dir")
        break
    done
    [ -n "$FIRST_JOB_ID" ] && break
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out waiting for first job after ${MAX_WAIT}s"
        exit 1
    fi
    echo "Waiting for job directory... (${WAITED}s)"
done

FIRST_JOB_DIR="$SLURM_HARNESS_DIR/jobs/$FIRST_JOB_ID"
echo "Found first job: ${FIRST_JOB_ID}"

# Wait for node_count to appear (reuse the same MAX_WAIT/WAITED)
while [ ! -f "$FIRST_JOB_DIR/node_count" ]; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out waiting for node_count in job ${FIRST_JOB_ID} after ${MAX_WAIT}s"
        exit 1
    fi
done
EXPECTED_WORKERS=$(cat "$FIRST_JOB_DIR/node_count")
echo "Discovered worker count: ${EXPECTED_WORKERS}"

echo "Waiting for ${EXPECTED_WORKERS} worker(s) to register..."
while true; do
    if job_workers_ready "$FIRST_JOB_DIR"; then
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out waiting for workers after ${MAX_WAIT}s"
        exit 1
    fi
    echo "Waiting for workers... (${WAITED}s elapsed)"
done

# Initial slurm.conf: header + first job's nodes
ALL_PARTITION_NODES=""
write_slurm_conf_header
add_job_nodes "$FIRST_JOB_ID"
start_slurmctld || exit 1

PROCESSED_JOBS=" ${FIRST_JOB_ID} "

echo "LOGIN_READY"

# --- Watcher loop: discover new job directories and add their nodes ---
# Unlike the previous approach that restarted slurmctld (which killed in-flight
# jobs), we append new NodeName entries to slurm.conf and run scontrol
# reconfigure. This preserves running jobs on existing nodes.
echo "Starting job watcher..."
while true; do
    for job_dir in "$SLURM_HARNESS_DIR/jobs"/*/; do
        [ -d "$job_dir" ] || continue
        job_id=$(basename "$job_dir")

        # Skip already-processed jobs
        case "$PROCESSED_JOBS" in
            *" ${job_id} "*) continue ;;
        esac

        # Check if all workers for this job are registered
        if ! job_workers_ready "$job_dir"; then
            continue
        fi

        echo "New job ${job_id}: adding workers..."
        add_job_nodes "$job_id"

        if scontrol reconfigure 2>/dev/null; then
            echo "slurmctld reconfigured for job ${job_id}"
        else
            echo "WARNING: scontrol reconfigure failed, falling back to slurmctld restart..."
            kill "$SLURMCTLD_PID" 2>/dev/null || true
            wait "$SLURMCTLD_PID" 2>/dev/null || true
            slurmctld -D &
            SLURMCTLD_PID=$!
            sleep 3
        fi

        PROCESSED_JOBS="${PROCESSED_JOBS}${job_id} "
    done
    sleep 10
done
