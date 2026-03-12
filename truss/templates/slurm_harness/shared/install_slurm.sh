#!/bin/bash
set -eux

# Common SLURM + munge installation for both login and worker nodes.
# This script is sourced (not executed) by setup_login.sh and setup_worker.sh.

SLURM_HARNESS_DIR="${BT_PROJECT_CACHE_DIR}/slurm_harness"
mkdir -p "$SLURM_HARNESS_DIR"

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq slurm-wlm slurm-client munge > /dev/null 2>&1

# Ensure munge directories exist with correct permissions
mkdir -p /etc/munge /var/log/munge /var/lib/munge /run/munge
chown -R munge:munge /etc/munge /var/log/munge /var/lib/munge /run/munge
chmod 700 /etc/munge
chmod 711 /run/munge

# Kill ALL munge processes — service stop may miss some
pkill -9 munged 2>/dev/null || true
service munge stop 2>/dev/null || true
# Remove the default munge key created by apt to avoid confusion
rm -f /etc/munge/munge.key

echo "SLURM and munge packages installed."
