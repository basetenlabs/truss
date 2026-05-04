# 02 — Raw Truss Reading Siblings

A regular Truss directory (not a `ChainletBase` subclass) that uses
`from truss_chains.runtime import get_service` to discover sibling URLs.
This is the bring_your_own_client pattern that Project 2 will rely on:
the Truss does not depend on the chains framework code, only on the
`truss_chains` package being installed.

The `plain_truss/` directory shows the full layout:
```
plain_truss/
├── config.yaml
└── model/
    └── model.py
```

`model.py` calls `get_service("Diarizer")` in `load()` and stashes the
URL in `predict()` — exactly how a TrussChainlet would surface sibling
information to user-managed RPC code.

## Run

```sh
uv run pytest test_raw_truss.py -v
```
