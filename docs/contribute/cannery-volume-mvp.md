# Cannery volume CLI MVP

`truss volume push|ls|show|pull` delegates volume operations to the Cannery
client binary. This remains a developer-facing vertical slice for RUN-871.
The wrapper has a verified artifact cache, but no customer-native Cannery
artifact is pinned until release automation publishes and reviews one.

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

This authentication path is temporary pending RUN-869. Truss does not exchange
a Baseten API token for a Cannery token yet. An explicit token file is required
for an explicit non-loopback endpoint. It must be a regular file owned and
readable only by the current user (`0600` on Unix). Only `localhost`,
`127.0.0.0/8`, and `::1` endpoints may run without a token. Truss passes a
token file to the child as `CANNERY_AUTH_TOKEN_FILE`; token values never appear
in process arguments. Truss forwards the selected organization as
`CANNERY_ORG`; Cannery versions that no longer need a separate organization
selector ignore it.

Without `TRUSS_CANNERY_API`, Truss maps the selected Baseten remote through an
explicit endpoint table. Unknown control-plane URLs fail closed pending the
final RUN-867 discovery contract. Remote execution also fails before starting
Cannery until a RUN-869 exchange adapter is configured.

## Local test flow

Build or obtain the matching Cannery client, then start a local Cannery server
and its object-store dependency as described in the Baseten BDN developer docs.
Point Truss at the client:

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

The subprocess contract is `cannery -o json --progress machine`: one final,
protocol-versioned JSON object on stdout, plus protocol-versioned NDJSON
progress events on stderr.
Use `--output json` on a Truss volume subcommand to preserve the final object on
stdout for scripts; progress remains on stderr.

> **Protocol migration TODO:** replace `Phase0ProtocolConsumer` with the
> generated Protobuf v1 ProtoJSON consumer after the schema branch lands. The
> swap occurs at `CanneryProtocolConsumer.start(...)`; command definitions,
> authentication, binary resolution, and subprocess supervision must not parse
> generated messages or change for that migration.

Truss creates a correlation ID before authentication or network access.
Failures retain an owner-only, redacted local diagnostic log and print its
path; successful commands remove the temporary log. Truss never uploads
diagnostics automatically.
