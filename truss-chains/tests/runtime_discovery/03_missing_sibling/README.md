# 03 — Missing Sibling (adversarial)

When user code calls `runtime.get_service("DoesNotExist")`, the framework
must raise `MissingDependencyError` with a message that lists the available
sibling names — not silently return `None` or crash with a `KeyError`.

This is the canonical failure-mode test for typos in chainlet names.

## Run

```sh
uv run pytest test_missing_sibling.py -v
```
