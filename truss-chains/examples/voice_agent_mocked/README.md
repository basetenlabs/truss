# voice_agent_mocked — Composable Chains centerpiece example

A CPU-only mock of the FDE voice-agent chain, designed for **fast iteration on chain-internal routing behavior** without paying for GPU model deploys. Same shape as `/Users/mattelim/Documents/fde/e2e-voice/`, but every chainlet is a few lines of Python returning canned data.

## What it shows

| Chainlet | Kind | Transport | Body |
|---|---|---|---|
| `STTMock` | `chains.TrussChainlet` | WebSocket | bytes → `{"text": "text-{N}"}` |
| `LLMMock` | `chains.TrussChainlet` | HTTP `predict` | `{"prompt"}` → `{"completion": "echo: ..."}` |
| `TTSMock` | `chains.TrussChainlet` | WebSocket | text → `text.encode()` |
| `Orchestrator` | `chains.ChainletBase` | WebSocket entrypoint | bytes in → drives STT→LLM→TTS → bytes out |

The orchestrator uses the same patterns as the FDE chain:

- GraphQL lookup of sibling oracle IDs (workaround until `dynamic_chainlet_config` exposes them — see `COMPOSABLE_CHAINS_PROPOSAL.md` "Discovered gap").
- Strip `HTTP_PROXY` / `HTTPS_PROXY` env vars at startup so outbound WS connections don't get intercepted.
- Build per-sibling URLs via `model-<oracle.id>.api.baseten.co/<env>/<endpoint>`.
- Use `httpx` (HTTP) and `websockets` (WS) with `Authorization: Api-Key`.

## Push

```sh
truss chains push --remote matte chain.py --watch
```

`--watch` keeps the orchestrator hot-patchable; it's the right mode for iterating on `chain.py` against a running chain.

A workspace secret named **`baseten_api_key`** must exist in your Baseten workspace; the chain reads it (declared via `Assets(secret_keys=["baseten_api_key"])`) for the GraphQL lookup. Set it once via the Baseten UI: <https://app.baseten.co/settings/secrets>.

## Invoke

After the push reports the entrypoint URL:

```sh
export BASETEN_API_KEY=...
export CHAIN_URL="wss://chain-<id>.api.baseten.co/development/websocket"
python client.py
```

Expected output:

```
reply (bytes, 15 bytes):
b'echo: text-1024'

✓ Match: b'echo: text-1024'
```

End-to-end the chain has done: send 1024 bytes → STT says `"text-1024"` → LLM says `"echo: text-1024"` → TTS encodes that as 15 bytes → returned to client.

## Why this exists

When iterating on **baseten-local** (chain gateway routing, operator VirtualService registration, dynamic config injection, promotion flow), the FDE chain is too slow a reproducer because its STT/LLM/TTS take 10–20 minutes per redeploy on H100s. This example reproduces the same chain shape — including the WS-fronted non-entrypoint chainlet pattern that surfaces the platform routing gap — in ≈2 minutes total deploy time.

Specifically, it's a fast probe for:

- **Issue 5/5b parity** (auto-uniquified `model_name` reaches the artifact, no name collisions on publish).
- **Issue 2 parity** (chain-internal `baseten_chain_api_key` auto-added to TrussChainlet artifacts).
- **The non-entrypoint WS routing gap** — does `wss://model-<oracle.id>.api.baseten.co/development/websocket` route to a non-entrypoint WS-fronted chainlet? If yes, the orchestrator's pipeline succeeds end-to-end. If no, the WS connect fails before LLM is reached.
- **Promotion flow** — `chains push --environment <env>` and Baseten-UI promotion against a tiny chain that completes in seconds.

## Cleanup

The chain stays cheap (CPU-only, scales to zero), so it's fine to leave running between iterations. To delete:

```sh
# Via the Baseten dashboard chain page → "Delete chain"
```
