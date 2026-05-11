#!/bin/bash
# Worker node (rank > 0) setup.
# Registers with the controller, starts slurmd, then idles.
# This script is sourced by setup_slurm.sh after install_slurm.sh.
#
# Expected variables from install_slurm.sh:
#   SLURM_DIR, ACTUAL_GPUS, WORKER_IP, WORKER_HOSTNAME

# Wait for controller to clean stale state
echo "Waiting for controller to initialize..."
while [ ! -f "$SLURM_DIR/ready_for_registration" ]; do sleep 2; done

# Register this node
mkdir -p "$SLURM_DIR/nodes"
echo "$WORKER_IP" > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_ip"
echo "$WORKER_HOSTNAME" > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_hostname"
echo "$ACTUAL_GPUS" > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_gpus"
grep -c ^processor /proc/cpuinfo > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_cpus"
echo "Node ${BT_NODE_RANK}: registered (${WORKER_HOSTNAME}, ${WORKER_IP}, ${ACTUAL_GPUS} GPUs)"

# Wait for munge key
echo "Waiting for munge key from controller..."
while [ ! -f "$SLURM_DIR/munge.key" ]; do sleep 2; done
cp "$SLURM_DIR/munge.key" /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key
service munge start || munged --force

# Wait for slurm.conf containing our hostname
echo "Waiting for slurm.conf..."
while true; do
    [ -f "$SLURM_DIR/slurm.conf" ] \
        && grep -q "$WORKER_HOSTNAME" "$SLURM_DIR/slurm.conf" 2>/dev/null \
        && break
    sleep 2
done

mkdir -p /var/spool/slurmd /var/log/slurm /etc/slurm
chmod 755 /var/spool/slurmd
cp "$SLURM_DIR/slurm.conf" /etc/slurm/slurm.conf

# Add all nodes to /etc/hosts
for i in $(seq 0 $((BT_GROUP_SIZE - 1))); do
    [ "$i" = "${BT_NODE_RANK}" ] && continue
    if [ -f "$SLURM_DIR/nodes/node_${i}_ip" ]; then
        OTHER_IP=$(cat "$SLURM_DIR/nodes/node_${i}_ip")
        OTHER_HOST=$(cat "$SLURM_DIR/nodes/node_${i}_hostname")
        echo "${OTHER_IP} ${OTHER_HOST}" >> /etc/hosts
    fi
done

configure_gres_and_cgroup

# Start slurmd
slurmd -D &
sleep 2
echo "Worker ${BT_NODE_RANK} slurmd started."

sleep infinity
