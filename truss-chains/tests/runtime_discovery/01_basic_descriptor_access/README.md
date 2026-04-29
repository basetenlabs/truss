# 01 — Basic Descriptor Access

A two-chainlet typed chain where `Caller` reads the new helper methods on the
`DeployedServiceDescriptor` returned by `context.get_service_descriptor("Echo")`
and reports them through `run_remote`.

This is the "typed chain that uses the new helpers" path — no raw Truss yet.
Demonstrates that `target_url`, `ws_url`, `internal_ws_url`, and
`with_auth_headers()` produce the expected shapes from a real descriptor.

## Run

```sh
uv run pytest test_descriptor_helpers.py -v
```
