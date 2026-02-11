#!/bin/bash
set -euo pipefail

# Manages nginx (auto-restart) and the model server (exit on failure) for docker_server.

NGINX_PID=""
MODEL_PID=""
SHUTDOWN=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

start_nginx() {
    log "Starting nginx..."
    nginx -g "daemon off;" &
    NGINX_PID=$!
}

start_model_server() {
    log "Starting model server: $START_COMMAND"
    bash -c "$START_COMMAND" &
    MODEL_PID=$!
}

wait_for_model_server() {
    log "Waiting for model server to be ready..."
    for _ in $(seq 1 30); do
        if ! kill -0 "$MODEL_PID" 2>/dev/null; then
            log "ERROR: Model server failed to start"
            return 1
        fi
        if curl -sf "http://127.0.0.1:${SERVER_PORT}${READINESS_ENDPOINT}" >/dev/null; then
            log "Model server is ready"
            return 0
        fi
        sleep 1
    done
    log "WARNING: Model server readiness check timed out, continuing"
    return 0
}

cleanup() {
    SHUTDOWN=true
    log "Stopping processes..."
    if [[ -n "$MODEL_PID" ]] && kill -0 "$MODEL_PID" 2>/dev/null; then
        kill -TERM "$MODEL_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$MODEL_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$MODEL_PID" 2>/dev/null || true
    fi
    if [[ -n "$NGINX_PID" ]] && kill -0 "$NGINX_PID" 2>/dev/null; then
        kill -TERM "$NGINX_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
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
if [[ -z "${READINESS_ENDPOINT:-}" ]]; then
    log "ERROR: READINESS_ENDPOINT environment variable is required"
    exit 1
fi

start_nginx
start_model_server
if ! wait_for_model_server; then
    exit 1
fi

log "Monitoring nginx and model server"

while [[ "$SHUTDOWN" == "false" ]]; do
    if ! kill -0 "$NGINX_PID" 2>/dev/null; then
        log "Nginx exited, restarting"
        start_nginx
    fi
    if ! kill -0 "$MODEL_PID" 2>/dev/null; then
        log "Model server exited, shutting down"
        exit 1
    fi
    sleep 1
done
