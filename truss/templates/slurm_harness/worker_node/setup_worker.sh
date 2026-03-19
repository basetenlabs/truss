#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_HARNESS_DIR="${BT_PROJECT_CACHE_DIR}/slurm_harness"

# Install SLURM and munge
source "${SCRIPT_DIR}/../shared/install_slurm.sh"

# Register this worker's info in shared cache BEFORE waiting for controller.
# The login node uses the job ID to compute our hostname via the Baseten naming
# convention: baseten-training-job-<job-id>-multinode-0[-N]
WORKER_IP=$(hostname -I | awk '{print $1}')
WORKER_HOSTNAME=$(hostname -s)
echo "$WORKER_IP" > "$SLURM_HARNESS_DIR/worker_${BT_NODE_RANK}_ip"
echo "$WORKER_HOSTNAME" > "$SLURM_HARNESS_DIR/worker_${BT_NODE_RANK}_hostname"
nproc > "$SLURM_HARNESS_DIR/worker_${BT_NODE_RANK}_cpus"
echo "Worker ${BT_NODE_RANK} CPU count: $(nproc)"

# Write job ID so the login node can compute all worker hostnames
# All nodes in this job share the same job ID, so any rank can write it
echo "$WORKER_HOSTNAME" | sed 's/baseten-training-job-\(.*\)-multinode-.*/\1/' > "$SLURM_HARNESS_DIR/worker_job_id"

# Write total node count so the login node knows how many workers to expect
echo "$EXPECTED_WORKERS" > "$SLURM_HARNESS_DIR/worker_node_count"

# Detect actual GPU devices and write count to shared cache
ACTUAL_GPUS=0
for dev in /dev/nvidia[0-9]*; do
    if [ -e "$dev" ]; then
        ACTUAL_GPUS=$((ACTUAL_GPUS + 1))
    fi
done
echo "$ACTUAL_GPUS" > "$SLURM_HARNESS_DIR/worker_${BT_NODE_RANK}_gpus"

echo "Worker ${BT_NODE_RANK} registered: hostname=${WORKER_HOSTNAME} ip=${WORKER_IP} gpus=${ACTUAL_GPUS}"

# Poll for controller IP (login node must be running)
echo "Waiting for controller node..."
MAX_WAIT=300
WAITED=0
while [ ! -f "$SLURM_HARNESS_DIR/controller_ip" ]; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out waiting for controller node after ${MAX_WAIT}s"
        exit 1
    fi
    echo "Waiting for controller IP... (${WAITED}s)"
done

CONTROLLER_IP=$(cat "$SLURM_HARNESS_DIR/controller_ip")
CONTROLLER_HOSTNAME=$(cat "$SLURM_HARNESS_DIR/controller_hostname")
echo "Found controller at ${CONTROLLER_IP} (${CONTROLLER_HOSTNAME})"

# Add controller to /etc/hosts
echo "${CONTROLLER_IP} ${CONTROLLER_HOSTNAME}" >> /etc/hosts

# Copy munge key from shared cache
cp "$SLURM_HARNESS_DIR/munge.key" /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

# Start munge
pkill -9 munged 2>/dev/null || true
sleep 1
service munge start || munged --force

# Wait for slurm.conf that contains THIS worker's hostname.
# The login node's watcher reconfigures slurmctld when it detects a new job ID,
# so we poll until slurm.conf includes our real hostname.
echo "Waiting for slurm.conf containing ${WORKER_HOSTNAME}..."
MAX_WAIT=300
WAITED=0
while true; do
    if [ -f "$SLURM_HARNESS_DIR/slurm.conf" ] && grep -q "$WORKER_HOSTNAME" "$SLURM_HARNESS_DIR/slurm.conf" 2>/dev/null; then
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out waiting for slurm.conf with our hostname after ${MAX_WAIT}s"
        exit 1
    fi
    echo "Waiting for slurm.conf with ${WORKER_HOSTNAME}... (${WAITED}s)"
done
echo "Found slurm.conf with our hostname"

# Copy slurm.conf from shared cache
cp "$SLURM_HARNESS_DIR/slurm.conf" /etc/slurm/slurm.conf

# Add all worker hostnames to /etc/hosts (for inter-worker communication)
EXPECTED_WORKERS="${EXPECTED_WORKERS:-1}"
for i in $(seq 0 $((EXPECTED_WORKERS - 1))); do
    if [ "$i" != "${BT_NODE_RANK}" ]; then
        if [ -f "$SLURM_HARNESS_DIR/worker_${i}_ip" ] && [ -f "$SLURM_HARNESS_DIR/worker_${i}_hostname" ]; then
            OTHER_IP=$(cat "$SLURM_HARNESS_DIR/worker_${i}_ip")
            OTHER_HOSTNAME=$(cat "$SLURM_HARNESS_DIR/worker_${i}_hostname")
            echo "${OTHER_IP} ${OTHER_HOSTNAME}" >> /etc/hosts
        fi
    fi
done

# Create required directories
mkdir -p /var/spool/slurmd /var/log/slurm
chmod 755 /var/spool/slurmd

# Configure gres (GPU resources) using the devices detected earlier
mkdir -p /etc/slurm
cat > /etc/slurm/gres.conf <<GRESCONF
# GPU resource definitions for this worker (auto-detected)
GRESCONF

for dev in /dev/nvidia[0-9]*; do
    if [ -e "$dev" ]; then
        echo "Name=gpu File=${dev}" >> /etc/slurm/gres.conf
    fi
done
echo "gres.conf: $(cat /etc/slurm/gres.conf | grep -c '^Name=gpu') GPU(s) configured"

# Start slurmd
slurmd -D &
SLURMD_PID=$!

sleep 3

if ! kill -0 $SLURMD_PID 2>/dev/null; then
    echo "ERROR: slurmd failed to start"
    cat /var/log/slurm/slurmd.log || true
    exit 1
fi

echo "Worker ${BT_NODE_RANK} slurmd started and connected to controller"

# Node 0 handles sbatch submission
if [ "${BT_NODE_RANK}" = "0" ]; then
    echo "Worker 0: Waiting for all ${EXPECTED_WORKERS} workers to register..."

    MAX_WAIT=600
    WAITED=0
    while true; do
        # Count idle/mixed workers visible to sinfo
        READY_COUNT=$(sinfo -N --noheader 2>/dev/null | grep -cE "idle|mixed|alloc" 2>/dev/null || true)
        READY_COUNT=$(echo "$READY_COUNT" | tr -d '[:space:]')
        READY_COUNT=${READY_COUNT:-0}
        echo "Workers ready: ${READY_COUNT}/${EXPECTED_WORKERS} (${WAITED}s elapsed)"

        if [ "$READY_COUNT" -ge "$EXPECTED_WORKERS" ]; then
            break
        fi

        sleep 10
        WAITED=$((WAITED + 10))
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "ERROR: Timed out waiting for workers after ${MAX_WAIT}s"
            sinfo || true
            exit 1
        fi
    done

    echo "WORKERS_READY"

    # Write the sbatch script from env var
    if [ -z "${SBATCH_SCRIPT:-}" ]; then
        echo "ERROR: SBATCH_SCRIPT environment variable is not set"
        exit 1
    fi

    SBATCH_SCRIPT_PATH="/tmp/sbatch_job.sh"
    echo "$SBATCH_SCRIPT" > "$SBATCH_SCRIPT_PATH"
    chmod +x "$SBATCH_SCRIPT_PATH"

    # Build explicit nodelist using real Baseten hostnames
    WORKER_JOB_ID=$(cat "$SLURM_HARNESS_DIR/worker_job_id")
    NODELIST=""
    for i in $(seq 0 $((EXPECTED_WORKERS - 1))); do
        NODE_HOSTNAME=$(cat "$SLURM_HARNESS_DIR/worker_${i}_hostname")
        if [ -z "$NODELIST" ]; then
            NODELIST="$NODE_HOSTNAME"
        else
            NODELIST="${NODELIST},${NODE_HOSTNAME}"
        fi
    done

    echo "Submitting sbatch job on nodes: ${NODELIST} (${EXPECTED_WORKERS} nodes)..."
    SBATCH_OUTPUT=$(sbatch --chdir="$(pwd)" --nodes="${EXPECTED_WORKERS}" --nodelist="${NODELIST}" "$SBATCH_SCRIPT_PATH" 2>&1)
    SBATCH_EXIT=$?

    if [ "$SBATCH_EXIT" -eq 0 ]; then
        SLURM_JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+$' || echo "unknown")
        echo "SBATCH_RESULT:${SLURM_JOB_ID}"
        echo "$SLURM_JOB_ID" > "$SLURM_HARNESS_DIR/slurm_job_id"
    else
        echo "SBATCH_ERROR:${SBATCH_OUTPUT}"
        exit 1
    fi

    # Stream SLURM job output to stdout so it appears in training logs.
    # Query scontrol for the actual output path since it depends on WorkDir.
    # We use tail -f with set +e to prevent the parent's set -e from killing
    # the subshell on benign non-zero exits (e.g. while-loop termination).
    SLURM_OUTPUT_FILE=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'StdOut=\K\S+' || echo "/slurm-${SLURM_JOB_ID}.out")
    echo "SLURM output file: ${SLURM_OUTPUT_FILE}"
    (
        set +e
        TWAIT=0
        while [ ! -f "$SLURM_OUTPUT_FILE" ]; do
            sleep 1
            TWAIT=$((TWAIT + 1))
            if [ "$TWAIT" -ge 300 ]; then break; fi
        done
        [ -f "$SLURM_OUTPUT_FILE" ] && exec tail -f "$SLURM_OUTPUT_FILE"
    ) &
    TAIL_PID=$!

    # Monitor the SLURM job until completion.
    # COMPLETING is a transient state where SLURM runs epilog — the job
    # itself has already finished, so we treat it as done.
    echo "Monitoring SLURM job ${SLURM_JOB_ID}..."
    while true; do
        JOB_STATE=$(squeue -j "$SLURM_JOB_ID" --noheader -o "%T" 2>/dev/null || echo "GONE")
        if [ "$JOB_STATE" = "GONE" ] || [ -z "$JOB_STATE" ] || [ "$JOB_STATE" = "COMPLETING" ] || [ "$JOB_STATE" = "COMPLETED" ]; then
            echo "SLURM job ${SLURM_JOB_ID} completed (state: ${JOB_STATE})"
            echo "SBATCH_COMPLETED:${SLURM_JOB_ID}"
            break
        fi
        sleep 15
    done

    # Stop tailing
    kill "$TAIL_PID" 2>/dev/null || true

    echo "Worker 0 exiting after job completion."
    exit 0
fi

# Non-zero workers: wait for the SLURM job to finish executing on this node,
# then exit. slurmd handles the job lifecycle — once squeue is empty, we're done.
while squeue --noheader 2>/dev/null | grep -q .; do
    sleep 15
done
echo "Worker ${BT_NODE_RANK} exiting, no remaining SLURM jobs."
exit 0
