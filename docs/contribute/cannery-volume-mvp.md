# Cannery volume CLI MVP

`truss volume push|ls|show|pull` delegates volume operations to the Cannery
client binary. The wrapper has a verified artifact cache, but no client artifact
is pinned until an immutable public artifact release is published and reviewed.

## Temporary configuration

The following environment variables bridge Truss to Cannery:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRUSS_CANNERY_BIN` | Explicit development/offline Cannery executable | unset |
| `TRUSS_CANNERY_API` | Explicit local Cannery API override | active Truss remote, or `http://127.0.0.1:8787` with no remote |
| `TRUSS_CANNERY_ORG` | Cannery organization | `dev` |
| `TRUSS_CANNERY_AUTH_TOKEN_FILE` | Existing owner-only Cannery bearer-token file | unset |
| `TRUSS_CANNERY_CACHE_DIR` | Verified binary cache override for tests | `~/.cache/truss/cannery` |
| `TRUSS_CANNERY_DIAGNOSTIC_DIR` | Redacted diagnostic-log directory override | `~/.cache/truss/cannery/diagnostics` |
| `TRUSS_CANNERY_PHASE0` | Set to `1` for an explicit loopback-only prototype protocol fallback | unset |

Production token exchange is not configured yet. An explicit token file is
required for an explicit non-loopback endpoint. It must be a regular file owned
and readable only by the current user (`0600` on Unix). Only `localhost`,
`127.0.0.0/8`, and `::1` endpoints may run without a token. Truss passes a token
file to the child as `CANNERY_AUTH_TOKEN_FILE`; token values never appear in
process arguments. Truss forwards the selected organization as `CANNERY_ORG`;
Cannery versions that no longer need a separate organization selector ignore
it.

Without `TRUSS_CANNERY_API`, Truss maps the selected Baseten remote to an
explicit public volume API endpoint. Unknown control-plane URLs fail closed
rather than guessing an endpoint. Remote execution also fails before starting
Cannery until a production token exchange adapter is configured.

## Local test flow

Build or obtain the matching Cannery client, then start a compatible local
server and its object-store dependency. Point Truss at the client:

```sh
export TRUSS_CANNERY_BIN=/path/to/cannery
export TRUSS_CANNERY_API=http://127.0.0.1:8787
export TRUSS_CANNERY_ORG=dev

uv run truss volume push ./weights bdn://default/weights:local
uv run truss volume ls default
uv run truss volume show bdn://default/weights:local
uv run truss volume pull bdn://default/weights:local ./downloaded
```

For a local server that enforces authentication, also set:

```sh
export TRUSS_CANNERY_AUTH_TOKEN_FILE=/path/to/cannery-token
chmod 600 /path/to/cannery-token
```

`TRUSS_CANNERY_BIN` and `cannery` on `PATH` are development paths. PATH lookup
is allowed only for loopback use. Remote use selects an exact platform artifact
whose URL, size, protocol version, and SHA-256 are trusted from the installed
Truss package. Downloads use Requests defaults, including `HTTPS_PROXY`,
`HTTP_PROXY`, `NO_PROXY`, and the normal corporate CA bundle mechanisms. TLS
verification cannot be disabled by this integration.

Truss first runs the non-mutating `cannery protocol` bootstrap and requires
machine protocol `1` with the `protojson-ndjson` encoding. The operation then
runs as `cannery --machine-protocol 1 ...`. Generated Protobuf types parse the
ordered stdout stream; typed progress, status, and warnings render on stderr,
and the typed terminal result becomes Truss text or `--output json`. Cannery
stderr is never parsed as protocol data. It is drained concurrently, bounded,
redacted, and retained only in failure diagnostics.

The generated schema, Python types, protocol documentation, and cross-language
golden fixtures are vendored under `truss/cli/cannery/generated`. Run
`scripts/sync_cannery_protocol_v1.py --check` to verify generated-code and hash
drift. Pass `--source-root` pointing at the canonical
`protocol/cannery/cli/v1` directory to compare with or synchronize from the
Baseten repository.

Prototype binaries that implement only the previous split stdout/stderr
contract remain available for loopback development by explicitly setting
`TRUSS_CANNERY_PHASE0=1`. Truss never automatically downgrades, and the fallback
is rejected for non-loopback endpoints and pinned production artifacts.

Truss creates a correlation ID before authentication or network access.
Failures retain an owner-only, redacted local diagnostic log and print its
path; successful commands remove the temporary log. Truss never uploads
diagnostics automatically.
