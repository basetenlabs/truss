#!/bin/sh

# SSH server startup script for inference model containers.
#
# Starts an OpenSSH server configured for CA-based certificate authentication.
# The SSH CA public key is mounted from a Kubernetes Secret. Users connect
# with CA-signed certificates whose principal must match BT_SSH_SUBJECT.
#
# This script is designed for non-root containers where the "app" user
# (uid 60000) already exists in the image. Unlike the training SSH script,
# it does not create users, modify /etc/passwd, or install packages at
# runtime. openssh-server must be pre-installed in the image.
#
# Environment variables (set by the operator template):
#   BT_SSH_CA_KEY_PATH   - Path to the SSH CA public key file
#   BT_SSH_SUBJECT       - Subject identifier (used as authorized principal)
#   BT_SSH_DIR           - Directory for SSH runtime files (default: /run/baseten-ssh)

_SSH_RESET="\033[0m"
_SSH_CYAN="\033[36m"
_SSH_GREEN="\033[32m"
_SSH_YELLOW="\033[33m"
_SSH_RED="\033[31m"

_ssh_log()     { printf "${_SSH_CYAN}[baseten-ssh]${_SSH_RESET} %b\n" "$1"; }
_ssh_log_ok()  { printf "${_SSH_CYAN}[baseten-ssh]${_SSH_RESET} ${_SSH_GREEN}%b${_SSH_RESET}\n" "$1"; }
_ssh_log_warn() { printf "${_SSH_CYAN}[baseten-ssh]${_SSH_RESET} ${_SSH_YELLOW}WARNING: %b${_SSH_RESET}\n" "$1"; }
_ssh_log_err() { printf "${_SSH_CYAN}[baseten-ssh]${_SSH_RESET} ${_SSH_RED}ERROR: %b${_SSH_RESET}\n" "$1"; }

start_ssh_server() {
    local ssh_port="${1:-2222}"
    local ca_key_path="${BT_SSH_CA_KEY_PATH:-/etc/ssh-ca/ca.pub}"
    local ssh_dir="${BT_SSH_DIR:-/run/baseten-ssh}"
    local subject="${BT_SSH_SUBJECT:-}"

    _ssh_log "Starting SSH server setup..."

    if [ -z "$subject" ]; then
        _ssh_log_err "BT_SSH_SUBJECT is not set. SSH server will not start."
        return 1
    fi

    if [ ! -f "$ca_key_path" ]; then
        _ssh_log_err "SSH CA public key not found at $ca_key_path. SSH server will not start."
        return 1
    fi

    if ! command -v sshd >/dev/null 2>&1; then
        _ssh_log_err "sshd not found. openssh-server must be installed in the image. SSH server will not start."
        return 1
    fi

    mkdir -p "$ssh_dir"

    if [ ! -f "$ssh_dir/bt_ssh_host_ed25519_key" ]; then
        _ssh_log "Generating host keys..."
        ssh-keygen -t ed25519 -f "$ssh_dir/bt_ssh_host_ed25519_key" -N "" -q
        ssh-keygen -t rsa -b 4096 -f "$ssh_dir/bt_ssh_host_rsa_key" -N "" -q
    fi

    mkdir -p /run/sshd

    echo "$subject" > "$ssh_dir/bt_authorized_principals"

    local sftp_server=""
    for path in /usr/lib/openssh/sftp-server /usr/lib/ssh/sftp-server /usr/libexec/openssh/sftp-server; do
        if [ -f "$path" ]; then
            sftp_server="$path"
            break
        fi
    done
    if [ -z "$sftp_server" ]; then
        _ssh_log_warn "sftp-server not found, SFTP will not be available"
    fi

    cat > "$ssh_dir/bt_sshd_config" << SSHD_EOF
Port $ssh_port
HostKey $ssh_dir/bt_ssh_host_ed25519_key
HostKey $ssh_dir/bt_ssh_host_rsa_key
PidFile $ssh_dir/bt_sshd.pid
TrustedUserCAKeys $ca_key_path
AuthorizedPrincipalsFile $ssh_dir/bt_authorized_principals
AllowUsers app
StrictModes no
PasswordAuthentication no
KbdInteractiveAuthentication no
PrintMotd no
AcceptEnv LANG LC_*
${sftp_server:+Subsystem sftp $sftp_server}
SSHD_EOF

    local sshd_bin
    sshd_bin=$(command -v sshd)

    _ssh_log "Starting SSH server on port $ssh_port (subject=$subject)..."
    "$sshd_bin" -f "$ssh_dir/bt_sshd_config" -D -e &
    BT_SSH_PID=$!
    _ssh_log_ok "sshd started (subject=$subject)"
}

start_ssh_server "$@"
