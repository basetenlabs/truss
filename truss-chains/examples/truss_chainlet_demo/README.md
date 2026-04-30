# TrussChainlet Demo — Composable Chains Project 2

A deployable, **CPU-only** chain proving that `chains.ChainletBase` and
`chains.TrussChainlet` coexist in one chain. The entrypoint depends on
**both** kinds at once and calls each via its native API in the same
`run_remote`.

## What this proves

Three artifacts in one chain push:

* **`Caller`** — `ChainletBase` entrypoint. Depends on both `Reverser` and
  `EchoTruss`. In a single `run_remote` call it:
  1. invokes `Reverser` via the framework's typed stub
     (`await self._reverser.run_remote(text)`), and
  2. invokes `EchoTruss` via raw `httpx`, using Project 1's descriptor
     helpers (`target_url` + `with_auth_headers`).
* **`Reverser`** — `ChainletBase` dep. Returns the input reversed.
  Codegen produces its `model.py` and a typed stub for callers.
* **`EchoTruss`** — `TrussChainlet` dep. Wraps `./echo_truss/` (a plain
  Truss directory) without any rewrite. The user's `model.py` is preserved
  byte-for-byte; only `model_metadata.chains_metadata` is added to the
  copied `config.yaml`.

If the framework wires both paths correctly, `match`-style verification:

```
input              = "hello"
via_chainletbase_reverser → "olleh"   # typed stub call worked
via_truss_chainlet_echo   → "HELLO"   # BYO-httpx call worked
```

Both inside one response. That's the guarantee of co-existence.

## Setup

```sh
# From the truss repo root
pip install -e .
```

`Caller` declares `httpx>=0.27` in its `pip_requirements`. The git-pinned
truss override in each chainlet's `pip_requirements` ships the
Composable Chains framework (Projects 1+2) into every pod. Drop those
overrides once Project 2 ships to PyPI.

## Push

```sh
truss chains push --remote matte truss-chains/examples/truss_chainlet_demo/chain.py
```

…or `--watch` for live patches as you iterate.

## Invoke

```sh
export BASETEN_API_KEY="..."
export CHAIN_URL="https://chain-<id>.api.baseten.co/environments/production/run_remote"
python truss-chains/examples/truss_chainlet_demo/invoke.py "hello"
```

Expected output:

```json
{
  "input": "hello",
  "via_chainletbase_reverser": "olleh",
  "via_truss_chainlet_echo": "HELLO",
  "echo_target_url": "https://...api.baseten.co/.../chainlet/.../run_remote"
}
```

The two response paths are different mechanisms hitting different chainlets
in the same chain:

* `via_chainletbase_reverser` came from `Reverser`'s code-gen'd `model.py`,
  reached via the framework stub.
* `via_truss_chainlet_echo` came from `echo_truss/model/model.py` (plain
  Truss code, *not* generated), reached via `httpx.post` against the URL
  from `desc.target_url`.

Both succeeded — that's the proof of TrussChainlet's BYO-client integration.

## File layout

```
truss_chainlet_demo/
├── README.md           # this file
├── chain.py            # Caller + Reverser + EchoTruss declarations
├── echo_truss/         # plain Truss directory — wrapped by EchoTruss
│   ├── config.yaml
│   └── model/model.py  # uppercase echo handler
└── invoke.py           # post-deploy invocation script
```

Note that `echo_truss/` looks exactly like a standalone `truss push`-able
directory. Project 2's promise is: any such directory can become a chain
member by adding **one class definition** in `chain.py`. No rewrite, no
shape change, no framework imports leaking into the Truss's `model.py`.
