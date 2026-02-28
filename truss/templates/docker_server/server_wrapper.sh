#!/bin/bash
set -euo pipefail

# Shell script to manage nginx and model server processes with auto-restart.
# Replaces supervisord for docker_server deployments.

# Global variables for process management
declare -A PROCESS_PIDS
declare -A RESTART_COUNTS
declare -A LAST_RESTART_TIME
declare -A PROCESS_START_TIME
SHUTDOWN_REQUESTED=false

# Configuration
MAX_RESTART_ATTEMPTS=3
RESTART_RESET_TIME=10  # Reset restart counter after 10 seconds of stable operation
FATAL_STATE_GRACE_PERIOD=5  # Wait 5 seconds before declaring fatal state
LINEAR_BACKOFF_INTERVAL=1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

start_nginx() {
    log "Starting nginx..."
    nginx -g "daemon off;" &
    local nginx_pid=$!
    sleep 1
    if kill -0 "$nginx_pid" 2>/dev/null; then
        PROCESS_PIDS["nginx"]=$nginx_pid
        PROCESS_START_TIME["nginx"]=$(date +%s)
        log "Nginx started successfully (PID: $nginx_pid)"
        return 0
    else
        log "ERROR: Failed to start nginx"
        return 1
    fi
}

start_model_server() {
    log "Starting model server with command: $START_COMMAND"
    eval "$START_COMMAND" &
    local model_pid=$!
    log "Waiting for model server to be ready..."
    for i in {1..30}; do
        if ! kill -0 "$model_pid" 2>/dev/null; then
            log "ERROR: Model server failed to start"
            return 1
        fi
        if curl -s -f "http://localhost:${SERVER_PORT}/ready" >/dev/null 2>&1; then
            log "Model server is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            log "WARNING: Model server readiness check timed out, continuing anyway"
        fi
        sleep 1
    done
    PROCESS_PIDS["model_server"]=$model_pid
    PROCESS_START_TIME["model_server"]=$(date +%s)
    log "Model server started successfully (PID: $model_pid)"
    return 0
}

should_reset_restart_counter() {
    local process_name=$1
    local current_time=$(date +%s)
    local start_time=${PROCESS_START_TIME[$process_name]:-0}
    if [[ $((current_time - start_time)) -gt $RESTART_RESET_TIME ]]; then
        return 0
    fi
    return 1
}

restart_process() {
    local process_name=$1
    if should_reset_restart_counter "$process_name"; then
        RESTART_COUNTS[$process_name]=0
        log "Resetting restart counter for $process_name (process ran stably for $RESTART_RESET_TIME seconds)"
    fi
    local restart_count=${RESTART_COUNTS[$process_name]:-0}
    if [[ $restart_count -ge $MAX_RESTART_ATTEMPTS ]]; then
        log "ERROR: $process_name has reached max restart limit ($MAX_RESTART_ATTEMPTS)"
        return 1
    fi
    local backoff_time=$((restart_count * LINEAR_BACKOFF_INTERVAL))
    log "Restarting $process_name (attempt $((restart_count + 1))/$MAX_RESTART_ATTEMPTS) after ${backoff_time}s backoff"
    sleep $backoff_time
    RESTART_COUNTS[$process_name]=$((restart_count + 1))
    LAST_RESTART_TIME[$process_name]=$(date +%s)
    if [[ "$process_name" == "nginx" ]]; then
        start_nginx
    elif [[ "$process_name" == "model_server" ]]; then
        start_model_server
    fi
    return $?
}

check_process() {
    local process_name=$1
    local pid=${PROCESS_PIDS[$process_name]:-}
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

handle_process_failure() {
    local process_name=$1
    if [[ "$SHUTDOWN_REQUESTED" == "true" ]]; then
        return 0
    fi
    log "WARNING: $process_name process has stopped"
    unset PROCESS_PIDS[$process_name]
    if restart_process "$process_name"; then
        log "Successfully restarted $process_name"
        return 0
    else
        log "ERROR: Failed to restart $process_name after max attempts"
        log "Waiting $FATAL_STATE_GRACE_PERIOD seconds before declaring fatal state..."
        sleep $FATAL_STATE_GRACE_PERIOD
        if [[ "$SHUTDOWN_REQUESTED" == "true" ]]; then
            return 0
        fi
        log "ERROR: $process_name has entered FATAL state (exhausted restart attempts)"
        return 1
    fi
}

cleanup() {
    log "Received shutdown signal, stopping processes..."
    SHUTDOWN_REQUESTED=true
    if check_process "model_server"; then
        local model_pid=${PROCESS_PIDS["model_server"]}
        log "Stopping model server (PID: $model_pid)"
        kill -TERM "$model_pid" 2>/dev/null || true
        for i in {1..30}; do
            if ! kill -0 "$model_pid" 2>/dev/null; then
                log "Model server stopped gracefully"
                break
            fi
            if [[ $i -eq 30 ]]; then
                log "Force killing model server"
                kill -KILL "$model_pid" 2>/dev/null || true
            fi
            sleep 1
        done
    fi
    if check_process "nginx"; then
        local nginx_pid=${PROCESS_PIDS["nginx"]}
        log "Stopping nginx (PID: $nginx_pid)"
        kill -TERM "$nginx_pid" 2>/dev/null || true
        for i in {1..10}; do
            if ! kill -0 "$nginx_pid" 2>/dev/null; then
                log "Nginx stopped gracefully"
                break
            fi
            if [[ $i -eq 10 ]]; then
                log "Force killing nginx"
                kill -KILL "$nginx_pid" 2>/dev/null || true
            fi
            sleep 1
        done
    fi
    log "Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

if [[ -z "${START_COMMAND:-}" ]]; then
    log "ERROR: START_COMMAND environment variable is required"
    exit 1
fi

if [[ -z "${SERVER_PORT:-}" ]]; then
    log "ERROR: SERVER_PORT environment variable is required"
    exit 1
fi

log "Starting custom server wrapper"
log "Model server command: $START_COMMAND"
log "Server port: $SERVER_PORT"

RESTART_COUNTS["nginx"]=0
RESTART_COUNTS["model_server"]=0

if ! start_nginx; then
    log "ERROR: Failed to start nginx initially"
    exit 1
fi

if ! start_model_server; then
    log "ERROR: Failed to start model server initially"
    cleanup
    exit 1
fi

log "Both services are running, monitoring processes..."

FATAL_STATE_REACHED=false

while [[ "$FATAL_STATE_REACHED" == "false" && "$SHUTDOWN_REQUESTED" == "false" ]]; do
    if ! check_process "nginx"; then
        log "WARNING: Nginx process has stopped"
        if ! handle_process_failure "nginx"; then
            log "ERROR: Nginx has entered FATAL state"
            FATAL_STATE_REACHED=true
        fi
    fi
    if [[ "$FATAL_STATE_REACHED" == "false" ]] && ! check_process "model_server"; then
        log "WARNING: Model server process has stopped"
        if ! handle_process_failure "model_server"; then
            log "ERROR: Model server has entered FATAL state"
            FATAL_STATE_REACHED=true
        fi
    fi
    sleep 5
done

if [[ "$FATAL_STATE_REACHED" == "true" ]]; then
    log "ERROR: One or more processes have entered FATAL state - shutting down"
    cleanup
    exit 1
fi

if [[ "$SHUTDOWN_REQUESTED" == "true" ]]; then
    cleanup
fi
