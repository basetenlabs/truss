#!/bin/bash
set -euo pipefail

# Enhanced shell script to match supervisord behavior for custom servers
# Manages nginx and model server processes with auto-restart and proper output handling

# Global variables for process management
declare -A PROCESS_PIDS
declare -A RESTART_COUNTS
declare -A LAST_RESTART_TIME
declare -A PROCESS_START_TIME
SHUTDOWN_REQUESTED=false

# Configuration matching supervisord defaults
MAX_RESTART_ATTEMPTS=3
RESTART_RESET_TIME=10  # Reset restart counter after 10 seconds of stable operation
FATAL_STATE_GRACE_PERIOD=5  # Wait 5 seconds before declaring fatal state
LINEAR_BACKOFF_INTERVAL=1  # supervisord uses linear backoff by default

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Function to start nginx with output redirection
start_nginx() {
    log "Starting nginx..."

    # Start nginx with output redirected to stdout (like supervisord's stdout_logfile=/dev/fd/1)
    nginx -g "daemon off;" &
    local nginx_pid=$!

    # Give nginx a moment to start
    sleep 1

    # Check if nginx started successfully
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

# Function to start model server with output redirection
start_model_server() {
    log "Starting model server with command: $START_COMMAND"

    # Start model server with output redirected to stdout (like supervisord)
    # Using eval to handle complex commands properly
    eval "$START_COMMAND" &
    local model_pid=$!

    # Wait for model server to be ready (similar to supervisord's startsecs=30)
    log "Waiting for model server to be ready..."
    for i in {1..30}; do
        if ! kill -0 "$model_pid" 2>/dev/null; then
            log "ERROR: Model server failed to start"
            return 1
        fi

        # Check if server is responding (basic health check)
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

# Function to check if restart counter should be reset (like supervisord's startsecs behavior)
should_reset_restart_counter() {
    local process_name=$1
    local current_time=$(date +%s)
    local start_time=${PROCESS_START_TIME[$process_name]:-0}

    # Reset counter if process has been running stably for RESTART_RESET_TIME seconds
    if [[ $((current_time - start_time)) -gt $RESTART_RESET_TIME ]]; then
        return 0
    fi
    return 1
}

# Function to restart a process with backoff (matching supervisord behavior)
restart_process() {
    local process_name=$1

    # Check if we should reset the restart counter (process ran successfully for a while)
    if should_reset_restart_counter "$process_name"; then
        RESTART_COUNTS[$process_name]=0
        log "Resetting restart counter for $process_name (process ran stably for $RESTART_RESET_TIME seconds)"
    fi

    local restart_count=${RESTART_COUNTS[$process_name]:-0}

    if [[ $restart_count -ge $MAX_RESTART_ATTEMPTS ]]; then
        log "ERROR: $process_name has reached max restart limit ($MAX_RESTART_ATTEMPTS)"
        return 1
    fi

    # Use linear backoff like supervisord (not exponential)
    local backoff_time=$((restart_count * LINEAR_BACKOFF_INTERVAL))

    log "Restarting $process_name (attempt $((restart_count + 1))/$MAX_RESTART_ATTEMPTS) after ${backoff_time}s backoff"
    sleep $backoff_time

    # Increment restart count and record restart time
    RESTART_COUNTS[$process_name]=$((restart_count + 1))
    LAST_RESTART_TIME[$process_name]=$(date +%s)

    # Restart the process
    if [[ "$process_name" == "nginx" ]]; then
        start_nginx
    elif [[ "$process_name" == "model_server" ]]; then
        start_model_server
    fi

    return $?
}

# Function to check if a process is running
check_process() {
    local process_name=$1
    local pid=${PROCESS_PIDS[$process_name]:-}

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Function to handle process failures (implements supervisord's PROCESS_STATE_FATAL behavior)
handle_process_failure() {
    local process_name=$1

    if [[ "$SHUTDOWN_REQUESTED" == "true" ]]; then
        return 0  # Don't restart if shutdown was requested
    fi

    log "WARNING: $process_name process has stopped"

    # Remove from process tracking
    unset PROCESS_PIDS[$process_name]

    # Attempt to restart the process
    if restart_process "$process_name"; then
        log "Successfully restarted $process_name"
        return 0
    else
        log "ERROR: Failed to restart $process_name after max attempts"

        # Implement supervisord's PROCESS_STATE_FATAL behavior - wait before declaring fatal
        log "Waiting $FATAL_STATE_GRACE_PERIOD seconds before declaring fatal state..."
        sleep $FATAL_STATE_GRACE_PERIOD

        # Check if shutdown was requested during grace period
        if [[ "$SHUTDOWN_REQUESTED" == "true" ]]; then
            return 0
        fi

        log "ERROR: $process_name has entered FATAL state (exhausted restart attempts)"
        return 1
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    log "Received shutdown signal, stopping processes..."
    SHUTDOWN_REQUESTED=true

    # Stop model server if running
    if check_process "model_server"; then
        local model_pid=${PROCESS_PIDS["model_server"]}
        log "Stopping model server (PID: $model_pid)"
        kill -TERM "$model_pid" 2>/dev/null || true

        # Wait for graceful shutdown (max 30 seconds)
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

    # Stop nginx if running
    if check_process "nginx"; then
        local nginx_pid=${PROCESS_PIDS["nginx"]}
        log "Stopping nginx (PID: $nginx_pid)"
        kill -TERM "$nginx_pid" 2>/dev/null || true

        # Wait for graceful shutdown (max 10 seconds)
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

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Validate required environment variables
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

# Initialize restart counts using associative arrays (properly scoped)
RESTART_COUNTS["nginx"]=0
RESTART_COUNTS["model_server"]=0

# Start both processes
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

# Main monitoring loop - matches supervisord's behavior
FATAL_STATE_REACHED=false

while [[ "$FATAL_STATE_REACHED" == "false" && "$SHUTDOWN_REQUESTED" == "false" ]]; do
    # Check nginx status
    if ! check_process "nginx"; then
        log "WARNING: Nginx process has stopped"
        if ! handle_process_failure "nginx"; then
            log "ERROR: Nginx has entered FATAL state"
            FATAL_STATE_REACHED=true
        fi
    fi

    # Check model server status (only if we haven't reached fatal state)
    if [[ "$FATAL_STATE_REACHED" == "false" ]] && ! check_process "model_server"; then
        log "WARNING: Model server process has stopped"
        if ! handle_process_failure "model_server"; then
            log "ERROR: Model server has entered FATAL state"
            FATAL_STATE_REACHED=true
        fi
    fi

    # Sleep for a short interval before checking again (like supervisord)
    sleep 5
done

# Handle fatal state (like supervisord's PROCESS_STATE_FATAL)
if [[ "$FATAL_STATE_REACHED" == "true" ]]; then
    log "ERROR: One or more processes have entered FATAL state - shutting down"
    cleanup
    exit 1
fi

# Normal shutdown
if [[ "$SHUTDOWN_REQUESTED" == "true" ]]; then
    cleanup
fi
