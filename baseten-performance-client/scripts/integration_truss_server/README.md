# Integration Truss Server

This directory contains a small custom-server Truss deployment for endpoint-pool
integration testing.

It is intentionally small:

- `GET /always_healthy`
- `GET /health`
- `POST /v1/embeddings`

The testing routes are controlled by UTC time windows from environment
variables so you can deploy multiple instances with staggered health and
serving behavior.

## What It Is For

This server is useful for testing endpoint routing, health checks, and failover
behavior against real Baseten deployments.

Example rollout:

- server A serves requests during `0-30` seconds of each minute
- server B serves requests during `30-60` seconds of each minute
- server A marks itself unhealthy slightly before it stops serving
- server B marks itself healthy slightly before it starts serving

That lets you verify:

- health checks move traffic away before requests start failing
- endpoint-pool failover keeps finding a healthy endpoint
- request routing does not produce user-visible `400` responses during handoff

## Directory Layout

- `config.yaml`: Truss config for `docker_server`
- `docker/`: Docker image source for the custom FastAPI server

## Build And Push The Docker Image

1. `cd scripts/integration_truss_server/docker`
2. update `VERSION`
3. build and push:

```bash
sh build_upload_new_image.sh
```

4. update `base_image.image` in [config.yaml](/Users/michaelfeil/work/truss/baseten-performance-client/scripts/integration_truss_server/config.yaml)

## Deploy With Truss

From this directory:

```bash
uv run truss push .
```

You can deploy the same Truss multiple times with different environment
variables in Baseten.

## Environment Variables

All windows are interpreted in UTC.

- `INTEGRATION_SERVER_NAME`
  Value exposed in response headers. Useful for debugging which deployment
  answered a request.
- `HEALTH_MINUTES_UTC`
  Minute-of-hour window set for `/health`. Default: `*`
- `HEALTH_SECONDS_UTC`
  Second-of-minute window set for `/health`. Default: `*`
- `SERVE_MINUTES_UTC`
  Minute-of-hour window set for `/v1/embeddings`. Default: `*`
- `SERVE_SECONDS_UTC`
  Second-of-minute window set for `/v1/embeddings`. Default: `*`
- `SERVE_GRACE_PERIOD_S`
  Continue accepting `/v1/embeddings` requests for this many seconds after the
  serve window turns inactive. Default: `0`
- `EMBEDDING_DIM`
  Fixed embedding dimension returned by the server. Default: `8`
- `RESPONSE_DELAY_MS`
  Optional artificial delay before returning responses. Default: `0`

Window format:

- `*` means always active
- `30-60` means start-inclusive, end-exclusive
- `0-25,40-60` means multiple ranges
- `17` means a single exact value

## Recommended Two-Deployment Setup

### Deployment A

- `INTEGRATION_SERVER_NAME=window-a`
- `HEALTH_SECONDS_UTC=0-25`
- `SERVE_SECONDS_UTC=0-30`
- `SERVE_GRACE_PERIOD_S=10`

### Deployment B

- `INTEGRATION_SERVER_NAME=window-b`
- `HEALTH_SECONDS_UTC=25-60`
- `SERVE_SECONDS_UTC=30-60`
- `SERVE_GRACE_PERIOD_S=10`

This gives you:

- A serves first half of the minute
- B serves second half of the minute
- health flips a bit earlier than serve handoff
- requests can still be accepted briefly while traffic drains away

That buffer is useful because health checks are not instantaneous.

## Server Behavior

### `GET /always_healthy`

This route is intended for Baseten liveness/readiness so the deployment stays
up even when the performance-client-visible `/health` route turns unhealthy.

- returns `200` unconditionally

### `GET /health`

- returns `200` when the health window is active
- returns `503` when the health window is inactive

### `POST /v1/embeddings`

- returns an OpenAI-compatible embeddings payload when the serve window is active
- continues returning embeddings for `SERVE_GRACE_PERIOD_S` after the serve
  window closes
- returns `400` with a message indicating the health-check protocol was violated
  when the serve window and grace period are both inactive

The server also returns debugging headers:

- `x-integration-server-name`
- `x-integration-server-minute-utc`
- `x-integration-server-second-utc`
- `x-integration-health-active`
- `x-integration-serve-active`
- `x-integration-serve-accepting`

## Local Run

From `docker/`:

```bash
docker build -t integration-truss-server:local .
docker run --rm -p 8000:8000 \
  -e INTEGRATION_SERVER_NAME=local \
  -e HEALTH_SECONDS_UTC='0-25' \
  -e SERVE_SECONDS_UTC='0-30' \
  integration-truss-server:local
```

Then:

```bash
curl -i localhost:8000/health
curl -i localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"test-model","input":["hello"]}'
```
