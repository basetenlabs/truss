#!/bin/bash
# Common SLURM + munge installation for all workstation nodes.
# This script is sourced (not executed) by setup_slurm.sh.

SLURM_DIR="${BT_PROJECT_CACHE_DIR}/slurm_workstation"
mkdir -p "$SLURM_DIR"

export DEBIAN_FRONTEND=noninteractive
echo "Installing SLURM packages..."
apt-get update -qq 2>&1
apt-get install -y -qq slurm-wlm slurm-client munge 2>&1

# Setup munge directories
mkdir -p /etc/munge /var/log/munge /var/lib/munge /run/munge
chown -R munge:munge /etc/munge /var/log/munge /var/lib/munge /run/munge
chmod 700 /etc/munge
chmod 711 /run/munge
pkill -9 munged 2>/dev/null || true
rm -f /etc/munge/munge.key

# Detect GPUs
ACTUAL_GPUS=0
for dev in /dev/nvidia[0-9]*; do
    [ -e "$dev" ] && ACTUAL_GPUS=$((ACTUAL_GPUS + 1))
done

WORKER_IP=$(hostname -I | awk '{print $1}')
WORKER_HOSTNAME=$(hostname -s)

echo "SLURM installed on ${WORKER_HOSTNAME} (${WORKER_IP}, ${ACTUAL_GPUS} GPUs)"

# Shared helper: configure gres.conf and cgroup.conf.
# Called by both controller and worker setup scripts.
configure_gres_and_cgroup() {
    mkdir -p /etc/slurm
    cat > /etc/slurm/gres.conf <<GRESCONF
GRESCONF
    for dev in /dev/nvidia[0-9]*; do
        [ -e "$dev" ] && echo "Name=gpu File=${dev}" >> /etc/slurm/gres.conf
    done

    # Use cgroup/v1 — containers have cgroup2 mounted read-only,
    # so the v2 plugin cannot create scopes.
    cat > /etc/slurm/cgroup.conf <<CGROUPCONF
CgroupPlugin=cgroup/v1
CGROUPCONF
}
