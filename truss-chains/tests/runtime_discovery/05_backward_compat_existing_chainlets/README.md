# 05 — Backward Compat: Existing Typed Chainlets (adversarial)

Project 1 internally refactors `populate_chainlet_service_predict_urls` to
share code with the new `truss_chains.runtime` API. This example pins the
historical contract of that refactor:

1. The function's signature and return shape are unchanged.
2. Error messages for "config absent" / "name missing" are preserved
   byte-for-byte.
3. The mutually-exclusive ``predict_url`` / ``internal_url`` behavior of
   the typed path is preserved: when ``internal_url`` is present in the
   dynamic config, ``predict_url`` is cleared on the resulting descriptor
   (matching the pre-refactor behavior).
4. Same Python type identity for `DeployedServiceDescriptor` regardless of
   import path.

## Run

```sh
uv run pytest test_backward_compat.py -v
```
