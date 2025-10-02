#!/bin/bash
set -euo pipefail

# Enhanced shell script to replace supervisord for custom servers
# Manages nginx and model server processes with proper signal handling

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

cleanup() {
    log "Received shutdown signal, stopping processes..."

    # Stop model server if running
    if [[ -n "${MODEL_SERVER_PID:-}" ]] && kill -0 "$MODEL_SERVER_PID" 2>/dev/null; then
        log "Stopping model server (PID: $MODEL_SERVER_PID)"
        kill -TERM "$MODEL_SERVER_PID" 2>/dev/null || true

        # Wait for graceful shutdown (max 30 seconds)
        for i in {1..30}; do
            if ! kill -0 "$MODEL_SERVER_PID" 2>/dev/null; then
                log "Model server stopped gracefully"
                break
            fi
            if [[ $i -eq 30 ]]; then
                log "Force killing model server"
                kill -KILL "$MODEL_SERVER_PID" 2>/dev/null || true
            fi
            sleep 1
        done
    fi

    # Stop nginx if running
    if [[ -n "${NGINX_PID:-}" ]] && kill -0 "$NGINX_PID" 2>/dev/null; then
        log "Stopping nginx (PID: $NGINX_PID)"
        kill -TERM "$NGINX_PID" 2>/dev/null || true

        # Wait for graceful shutdown (max 10 seconds)
        for i in {1..10}; do
            if ! kill -0 "$NGINX_PID" 2>/dev/null; then
                log "Nginx stopped gracefully"
                break
            fi
            if [[ $i -eq 10 ]]; then
                log "Force killing nginx"
                kill -KILL "$NGINX_PID" 2>/dev/null || true
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

# Start nginx in background (will run in foreground later)
log "Starting nginx..."
nginx -g "daemon off;" &
NGINX_PID=$!

# Wait a moment for nginx to start
sleep 2

# Verify nginx started successfully
if ! kill -0 "$NGINX_PID" 2>/dev/null; then
    log "ERROR: Failed to start nginx"
    exit 1
fi

log "Nginx started successfully (PID: $NGINX_PID)"

# Start model server in background
log "Starting model server..."
eval "$START_COMMAND" &
MODEL_SERVER_PID=$!

# Wait for model server to be ready (similar to supervisord's startsecs=30)
log "Waiting for model server to be ready..."
for i in {1..30}; do
    if ! kill -0 "$MODEL_SERVER_PID" 2>/dev/null; then
        log "ERROR: Model server failed to start"
        cleanup
        exit 1
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

# Monitor processes
log "Both services are running, monitoring processes..."
while true; do
    # Check if nginx is still running
    if ! kill -0 "$NGINX_PID" 2>/dev/null; then
        log "ERROR: Nginx process died"
        cleanup
        exit 1
    fi

    # Check if model server is still running
    if ! kill -0 "$MODEL_SERVER_PID" 2>/dev/null; then
        log "ERROR: Model server process died"
        cleanup
        exit 1
    fi

    # Sleep for a short interval before checking again
    sleep 5
done
