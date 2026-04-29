# 04 — Dynamic Config Absent (adversarial)

When a Truss runs outside any chain context (no
`/etc/b10_dynamic_config/dynamic_chainlet_config` file present):

- `runtime.list_services()` returns an empty mapping (does NOT raise) — so
  caller code can branch on "am I inside a chain?" without try/except.
- `runtime.get_service(name)` raises `MissingDependencyError` with a clear
  "not running inside a chain context" message.

This documents the contract for code that needs to behave gracefully both
inside and outside a chain (e.g., a Truss that's reused across deployments).

## Run

```sh
uv run pytest test_no_context.py -v
```
