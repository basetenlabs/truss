#!/bin/bash

# Start nginx and the model server; exit if either stops so the platform can restart.

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

cleanup() {
    log "Shutting down..."
    kill -TERM "$NGINX_PID" "$MODEL_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

log "Starting nginx"
nginx -g "daemon off;" &
NGINX_PID=$!

log "Starting model server: $START_COMMAND"
bash -c "$START_COMMAND" &
MODEL_PID=$!

log "Waiting for model server to be ready"
for _ in $(seq 1 30); do
    kill -0 "$MODEL_PID" 2>/dev/null || { log "Model server failed to start"; exit 1; }
    if curl -sf "http://127.0.0.1:${SERVER_PORT}${READINESS_ENDPOINT}" >/dev/null; then
        log "Model server is ready"
        break
    fi
    sleep 1
done

log "Running — container will exit if nginx or model server stops"
wait -n
log "A process exited, stopping container"
exit 1
