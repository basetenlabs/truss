#!/bin/bash
# Controller node (rank 0) setup.
# Runs slurmctld + slurmd, generates slurm.conf, then idles.
# This script is sourced by setup_slurm.sh after install_slurm.sh.
#
# Expected variables from install_slurm.sh:
#   SLURM_DIR, ACTUAL_GPUS, WORKER_IP, WORKER_HOSTNAME

# Clean stale state from previous runs before any workers register
rm -rf "$SLURM_DIR/nodes"
rm -f "$SLURM_DIR/slurm.conf" "$SLURM_DIR/munge.key"
mkdir -p "$SLURM_DIR/nodes"
# Signal that cleanup is done and workers can register
touch "$SLURM_DIR/ready_for_registration"

# Register this node
echo "$WORKER_IP" > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_ip"
echo "$WORKER_HOSTNAME" > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_hostname"
echo "$ACTUAL_GPUS" > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_gpus"
grep -c ^processor /proc/cpuinfo > "$SLURM_DIR/nodes/node_${BT_NODE_RANK}_cpus"
echo "Node ${BT_NODE_RANK}: registered (${WORKER_HOSTNAME}, ${WORKER_IP}, ${ACTUAL_GPUS} GPUs)"

# Generate and share munge key
dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key 2>/dev/null
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key
cp /etc/munge/munge.key "$SLURM_DIR/munge.key.tmp"
mv "$SLURM_DIR/munge.key.tmp" "$SLURM_DIR/munge.key"

# Start munge
service munge start || munged --force

# Wait for all nodes to register
echo "Waiting for ${BT_GROUP_SIZE} nodes to register..."
while true; do
    REGISTERED=$(ls "$SLURM_DIR/nodes"/node_*_ip 2>/dev/null | wc -l)
    [ "$REGISTERED" -ge "$BT_GROUP_SIZE" ] && break
    echo "  ${REGISTERED}/${BT_GROUP_SIZE} nodes registered..."
    sleep 3
done
echo "All ${BT_GROUP_SIZE} nodes registered."

# Build /etc/hosts and slurm.conf
CONTROLLER_IP="$WORKER_IP"
CONTROLLER_HOSTNAME="$WORKER_HOSTNAME"

mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /etc/slurm
chmod 755 /var/spool/slurmctld /var/spool/slurmd

cat > /etc/slurm/slurm.conf <<SLURMCONF
ClusterName=workstation
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
TaskPlugin=task/none
SlurmctldDebug=info
SlurmdDebug=info
SlurmdParameters=config_overrides
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
SlurmctldTimeout=120
WaitTime=0
GresTypes=gpu
SLURMCONF

ALL_NODES=""
for i in $(seq 0 $((BT_GROUP_SIZE - 1))); do
    NODE_IP=$(cat "$SLURM_DIR/nodes/node_${i}_ip")
    NODE_HOST=$(cat "$SLURM_DIR/nodes/node_${i}_hostname")
    NODE_GPUS=$(cat "$SLURM_DIR/nodes/node_${i}_gpus")
    NODE_CPUS=$(cat "$SLURM_DIR/nodes/node_${i}_cpus")

    # Add to /etc/hosts
    sed -i "/ ${NODE_HOST}$/d" /etc/hosts 2>/dev/null || true
    echo "${NODE_IP} ${NODE_HOST}" >> /etc/hosts

    echo "NodeName=${NODE_HOST} NodeAddr=${NODE_IP} CPUs=${NODE_CPUS} RealMemory=100000 Gres=gpu:${NODE_GPUS} State=UNKNOWN" >> /etc/slurm/slurm.conf

    [ -z "$ALL_NODES" ] && ALL_NODES="$NODE_HOST" || ALL_NODES="${ALL_NODES},${NODE_HOST}"
done

echo "PartitionName=gpu Nodes=${ALL_NODES} Default=YES MaxTime=INFINITE State=UP" >> /etc/slurm/slurm.conf

# Share slurm.conf with workers
cp /etc/slurm/slurm.conf "$SLURM_DIR/slurm.conf"

configure_gres_and_cgroup

# Clear stale state and start slurmctld
rm -f /var/spool/slurmctld/node_state /var/spool/slurmctld/node_state.old
rm -f /var/spool/slurmctld/job_state /var/spool/slurmctld/job_state.old
slurmctld -D &
sleep 3
echo "slurmctld started."

# Also run slurmd on node 0 (it's a worker too)
slurmd -D &
sleep 2
echo "slurmd started on controller node."

# Wait for all workers to show up in sinfo
echo "Waiting for all ${BT_GROUP_SIZE} nodes in SLURM..."
WAITED=0
while true; do
    READY=$(sinfo -N --noheader 2>/dev/null | grep -cE "idle|mixed|alloc" || true)
    [ "$READY" -ge "$BT_GROUP_SIZE" ] && break
    sleep 5
    WAITED=$((WAITED + 5))
    [ "$WAITED" -ge 300 ] && echo "WARNING: timeout waiting for SLURM nodes" && break
    echo "  SLURM nodes ready: ${READY}/${BT_GROUP_SIZE}"
done

echo ""
echo "============================================"
echo "  SLURM CLUSTER READY"
echo "  Nodes: ${BT_GROUP_SIZE}"
echo "  GPUs per node: ${ACTUAL_GPUS}"
echo "  Total GPUs: $((BT_GROUP_SIZE * ACTUAL_GPUS))"
echo "============================================"
echo ""
sinfo
echo ""

sleep infinity
