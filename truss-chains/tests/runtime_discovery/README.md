# Runtime Discovery — Contract Test Scenarios

CPU-only pytest scenarios that pin the Project 1 (Runtime Discovery Platform
Contract) API. Each scenario doubles as documentation of the expected shape
and behavior of `truss_chains.runtime`.

These live under `tests/` rather than `examples/` because they are **not
deployable chains** — they exercise the library by writing fake
`/etc/b10_dynamic_config/dynamic_chainlet_config` files via pytest
monkeypatch, not by pushing real chainlets to Baseten. (For deployable
chain examples, see `truss-chains/examples/`.)

## Setup

```sh
pip install -e /Users/mattelim/Documents/truss
pip install -e /Users/mattelim/Documents/truss/truss-chains
pip install pytest
```

## Run all

```sh
# Runs each scenario in its own pytest invocation, with descriptive headers.
bash run_all_local.sh

# Or as one combined pytest run:
uv run pytest truss-chains/tests/runtime_discovery/
```

## Per-scenario

| Dir | What it demonstrates | Type |
|---|---|---|
| `basic_descriptor_access/` | Typed chainlets accessing the new `target_url`, `ws_url`, `internal_ws_url`, `with_auth_headers()` helpers via `context.get_service_descriptor(...)`. | Positive |
| `02_raw_truss_reading_siblings/` | A non-`ChainletBase` Truss using `from truss_chains.runtime import get_service` to discover sibling URLs. | Positive |
| `03_missing_sibling/` | `MissingDependencyError` with helpful "Available: ..." message when looking up a non-existent sibling. | Adversarial |
| `04_dynamic_config_absent/` | `list_services()` returns `{}` (does not raise) and `get_service(...)` raises with a clear "not running inside a chain context" message when no config file is present. | Adversarial |
| `05_backward_compat_existing_chainlets/` | `populate_chainlet_service_predict_urls` (the typed-chain code path) preserves its historical mutually-exclusive `predict_url` / `internal_url` behavior — proves the internal refactor doesn't change the typed path. | Adversarial |

Each scenario has a `test_<name>.py` that runs under `pytest`. Scenarios that
involve a Truss directory have a `plain_truss/` subdirectory with a hand-written
`config.yaml` and `model/model.py` showing the bring_your_own_client pattern.
