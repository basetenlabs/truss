#!/bin/bash
set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_HARNESS_DIR="${BT_PROJECT_CACHE_DIR}/slurm_harness"

# Clean up all stale state from previous runs.
rm -f "$SLURM_HARNESS_DIR"/controller_* \
      "$SLURM_HARNESS_DIR"/slurm.conf \
      "$SLURM_HARNESS_DIR"/munge.key \
      "$SLURM_HARNESS_DIR"/munge_test_cred \
      "$SLURM_HARNESS_DIR"/worker_*_ip \
      "$SLURM_HARNESS_DIR"/worker_*_hostname \
      "$SLURM_HARNESS_DIR"/worker_*_cpus \
      "$SLURM_HARNESS_DIR"/worker_job_id

# Install SLURM and munge
source "${SCRIPT_DIR}/../shared/install_slurm.sh"

# Configure truss remote so workers can be pushed from the login node
if [ -n "${BASETEN_API_KEY:-}" ]; then
    python3 -c "import truss; truss.login('${BASETEN_API_KEY}')"
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

EXPECTED_WORKERS="${EXPECTED_WORKERS:-1}"
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

# Generate slurm.conf and update /etc/hosts for the given worker job ID.
# Writes to /etc/slurm/slurm.conf and copies to shared cache.
generate_slurm_conf() {
    local worker_job_id="$1"

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

    local partition_nodes=""
    for i in $(seq 0 $((EXPECTED_WORKERS - 1))); do
        local worker_ip
        worker_ip=$(cat "$SLURM_HARNESS_DIR/worker_${i}_ip")
        local worker_host
        worker_host=$(worker_hostname_for_rank "$worker_job_id" "$i")

        # Update /etc/hosts (remove stale entry first)
        sed -i "/ ${worker_host}$/d" /etc/hosts 2>/dev/null || true
        echo "${worker_ip} ${worker_host}" >> /etc/hosts

        # Read actual CPU count from worker's report, or default to 224 (H200)
        local worker_cpus=224
        if [ -f "$SLURM_HARNESS_DIR/worker_${i}_cpus" ]; then
            worker_cpus=$(cat "$SLURM_HARNESS_DIR/worker_${i}_cpus")
        fi
        echo "NodeName=${worker_host} NodeAddr=${worker_ip} CPUs=${worker_cpus} RealMemory=100000 Gres=gpu:${GPUS_PER_NODE} State=UNKNOWN" >> /etc/slurm/slurm.conf

        if [ -z "$partition_nodes" ]; then
            partition_nodes="$worker_host"
        else
            partition_nodes="${partition_nodes},${worker_host}"
        fi
    done

    echo "PartitionName=gpu Nodes=${partition_nodes} Default=YES MaxTime=INFINITE State=UP" >> /etc/slurm/slurm.conf
    cp /etc/slurm/slurm.conf "$SLURM_HARNESS_DIR/slurm.conf"
}

# Start (or restart) slurmctld. Clears stale state on restart.
start_slurmctld() {
    if [ -n "${SLURMCTLD_PID:-}" ]; then
        echo "Stopping slurmctld (PID ${SLURMCTLD_PID})..."
        kill "$SLURMCTLD_PID" 2>/dev/null || true
        wait "$SLURMCTLD_PID" 2>/dev/null || true
        rm -f /var/spool/slurmctld/node_state /var/spool/slurmctld/node_state.old
        rm -f /var/spool/slurmctld/job_state /var/spool/slurmctld/job_state.old
    fi

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

# Self-test: push a worker job from this login node before waiting for workers.
# Reads config from runtime_config.json (already on disk from the push).
if [ -f /workspace/runtime_config.json ]; then
    SELF_TEST=$(python3 -c "import json; print(json.load(open('/workspace/runtime_config.json')).get('self_test', False))" 2>/dev/null || echo "False")
    if [ "$SELF_TEST" = "True" ]; then
        PROJECT=$(python3 -c "import json; print(json.load(open('/workspace/runtime_config.json')).get('project_name', 'slurm-harness'))")
        PARTITION=$(python3 -c "import json; print(json.load(open('/workspace/runtime_config.json')).get('partition', 'H200'))")
        echo "SELF_TEST: Pushing a worker job from login node..."
        truss train slurm sbatch --wrap "echo SELF_TEST_OK && hostname && nvidia-smi -L && sleep 30 && echo SELF_TEST_DONE" \
            --project "$PROJECT" -p "$PARTITION" --gres "gpu:${GPUS_PER_NODE}" -N "${EXPECTED_WORKERS}" 2>&1 \
            || echo "SELF_TEST: sbatch failed with exit code $?"
        echo "SELF_TEST: Worker push initiated, now waiting for it to register..."
    fi
fi

# Wait for worker job ID and all worker IPs to appear in shared cache.
echo "Waiting for ${EXPECTED_WORKERS} worker(s) to register..."
MAX_WAIT=600
WAITED=0
while true; do
    REGISTERED=0
    for i in $(seq 0 $((EXPECTED_WORKERS - 1))); do
        if [ -f "$SLURM_HARNESS_DIR/worker_${i}_ip" ] && [ -f "$SLURM_HARNESS_DIR/worker_job_id" ]; then
            REGISTERED=$((REGISTERED + 1))
        fi
    done
    echo "Workers registered: ${REGISTERED}/${EXPECTED_WORKERS} (${WAITED}s elapsed)"
    if [ "$REGISTERED" -ge "$EXPECTED_WORKERS" ]; then
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out waiting for workers after ${MAX_WAIT}s"
        exit 1
    fi
done

WORKER_JOB_ID=$(cat "$SLURM_HARNESS_DIR/worker_job_id")
echo "Worker job ID: ${WORKER_JOB_ID}"

generate_slurm_conf "$WORKER_JOB_ID"
start_slurmctld || exit 1

echo "LOGIN_READY"

# Watcher loop: detect new worker jobs and restart slurmctld.
# SLURM does not allow node name changes via scontrol reconfigure —
# we must kill and restart slurmctld entirely.
echo "Starting worker watcher..."
while true; do
    if [ -f "$SLURM_HARNESS_DIR/worker_job_id" ]; then
        CURRENT_JOB_ID=$(cat "$SLURM_HARNESS_DIR/worker_job_id")
        if [ "$CURRENT_JOB_ID" != "$WORKER_JOB_ID" ]; then
            echo "New worker job detected: ${CURRENT_JOB_ID} (was ${WORKER_JOB_ID})"

            # Check if all worker IPs are available before reconfiguring.
            ALL_READY=true
            for i in $(seq 0 $((EXPECTED_WORKERS - 1))); do
                if [ ! -f "$SLURM_HARNESS_DIR/worker_${i}_ip" ]; then
                    ALL_READY=false
                    break
                fi
            done

            if [ "$ALL_READY" = true ]; then
                # Only update WORKER_JOB_ID after successful reconfiguration
                generate_slurm_conf "$CURRENT_JOB_ID"
                if start_slurmctld; then
                    WORKER_JOB_ID="$CURRENT_JOB_ID"
                    echo "Reconfigured for job ${WORKER_JOB_ID}"
                fi
            else
                echo "Not all workers ready yet, will retry..."
            fi
        fi
    fi
    sleep 10
done
