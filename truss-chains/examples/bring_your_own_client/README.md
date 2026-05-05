# Bring-Your-Own-Client — Composable Chains Project 1 Demo

A minimal, deployable, **CPU-only** chain that demonstrates what Project 1
(Runtime Discovery Platform Contract) actually unlocks at runtime.

## What this proves

Two chainlets:

- `Echo` — trivial dependency. `run_remote(text: str) -> str` returns the
  uppercased input.
- `Caller` — entrypoint. Calls `Echo` **two different ways** in the same
  request:
  1. Via the framework's auto-generated `StubBase` (the existing path).
  2. Via raw `httpx.AsyncClient`, using the new `DeployedServiceDescriptor`
     helpers `target_url` and `with_auth_headers(api_key)` to source the URL
     and auth from the chain context — without touching `StubBase`.

If Project 1's helpers correctly mirror what `BasetenSession.__init__` would
construct internally, both paths return the same string. The chain returns
both results plus the descriptor metadata so you can see them line up.

This is the **bring_your_own_client** pattern: keep the chain's UI grouping,
atomic deploy, and per-pod sibling-URL discovery, but plug in any HTTP / WS
client you like (httpx, websockets, grpc.aio, …).

## Setup

```sh
# From the truss repo root
pip install -e .
pip install -e truss-chains
truss login   # if not already authenticated
```

## Push

```sh
chains push truss-chains/examples/bring_your_own_client/chain.py
```

The CLI prints the chain's invoke URL when the deploy completes. Copy the
`run_remote` URL for the next step.

## Invoke

```sh
export BASETEN_API_KEY="..."
export CHAIN_URL="https://chain-<id>.api.baseten.co/environments/production/run_remote"
python truss-chains/examples/bring_your_own_client/invoke.py "hello world"
```

Expected output:

```json
{
  "via_stub": "HELLO WORLD",
  "via_byo_httpx": "HELLO WORLD",
  "match": true,
  "target_url": "https://...api.baseten.co/.../chainlet/.../run_remote",
  "has_internal_url": true,
  "auth_header_keys": ["Authorization", "Host"]
}

✓ MATCH
```

The interesting fields:

- `match: true` — proves the BYO-httpx call hit the same endpoint and got the
  same answer as the framework stub.
- `target_url` — what `desc.target_url` resolved to inside the chainlet pod.
  Should be the cluster-internal beefeater URL when `internal_url` is present
  (lower latency than going through the public chain hostname).
- `has_internal_url: true` — confirms the operator injected an `internal_url`
  alongside the public `predict_url`.
- `auth_header_keys: ["Authorization", "Host"]` — confirms `with_auth_headers`
  produced both the API key header and the chain hostname `Host:` override
  (the latter only appears when `internal_url` is set).

## Local smoke

You can run `chain.py` directly under `run_local` to inspect the descriptor
helpers without deploying:

```sh
python truss-chains/examples/bring_your_own_client/chain.py
```

Prints the four helper outputs against a fake `Echo` descriptor. Useful for
sanity-checking the helpers in development.

## What this does *not* show

- The `truss_chains.runtime.get_service(name)` API for **non-`ChainletBase`**
  Trusses. That use case is unlocked by Project 2 (`TrussChainlet`); see
  `test_plain_truss_picks_up_siblings` in `truss-chains/tests/test_runtime.py`
  (with the fixture under `tests/runtime_discovery/plain_truss/`) for the
  contract test.
- WebSocket-flavored siblings. The helpers `ws_url` / `internal_ws_url` are
  exercised in the runtime_discovery contract tests; a WebSocket-fronted
  deployable demo would need a chainlet using
  `truss_config.WebsocketOptions` plus a client that handles WS framing.
