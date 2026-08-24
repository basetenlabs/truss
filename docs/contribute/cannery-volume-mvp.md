# Cannery volume CLI MVP

`truss volume push|ls|show|pull` delegates volume operations to the Cannery
client binary. The wrapper has a verified artifact cache, but no client artifact
is pinned until an immutable public artifact release is published and reviewed.

## Temporary configuration

The following environment variables bridge Truss to Cannery:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRUSS_CANNERY_BIN` | Explicit development/offline Cannery executable | unset |
| `TRUSS_CANNERY_API` | Explicit loopback Cannery API for local development | unset |
| `TRUSS_CANNERY_ALLOW_PATH` | Set to `1` to explicitly execute `cannery` from `PATH` in loopback development | unset |
| `TRUSS_CANNERY_ORG` | Cannery organization | `dev` |
| `TRUSS_CANNERY_AUTH_TOKEN_FILE` | Existing owner-only Cannery bearer-token file | unset |
| `TRUSS_CANNERY_CACHE_DIR` | Verified binary cache override for tests | `~/.cache/truss/cannery` |
| `TRUSS_CANNERY_DIAGNOSTIC_DIR` | Redacted diagnostic-log directory override | `~/.cache/truss/cannery/diagnostics` |
| `TRUSS_CANNERY_PHASE0` | Set to `1` for an explicit loopback-only prototype protocol fallback | unset |

Production token exchange is not configured yet. The integration seam accepts
an exchanged token and optional absolute expiry. When an adapter supplies an
expiry, Truss refreshes before expiry in a bounded background lifecycle,
atomically replaces the token file, and stops and joins refresh during success,
failure, or cancellation. An explicit token file must be a regular file owned
and readable only by the current user (`0600` on Unix). Only an explicitly
configured `localhost`, `127.0.0.0/8`, or `::1` endpoint may run without a
token. Truss passes a token file to the child as `CANNERY_AUTH_TOKEN_FILE`;
token values never appear in process arguments.

Without `TRUSS_CANNERY_API`, Truss requires a configured Baseten remote. Pass
`--remote NAME` to any volume command when more than one is configured. Truss
maps the selected remote to an explicit public volume API endpoint; unknown
control-plane URLs fail closed rather than guessing an endpoint. Remote
execution also fails before starting Cannery until a production token exchange
adapter is configured.

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

To pull a subset, repeat `--include` in the desired order:

```sh
uv run truss volume pull bdn://default/weights:local ./downloaded \
  --include model.safetensors \
  --include tokenizer/
```

With no `--include`, Cannery pulls the complete volume. Includes are portable
volume-relative file or directory paths, not globs; repeated selectors form a
union. Cannery is the normative validator and reports invalid or unmatched
selectors as structured usage errors. Pull results report logical bytes and
file/directory counts for the selected graph. Subset results additionally
include the corresponding complete-volume totals.

Pulls resume automatically when matching saved state exists; no resume flag is
needed. Resume identity includes the resolved manifest digest, absolute
destination, and normalized selector set. Selector order and duplicates do not
change that identity, while a moved tag or different selector set cannot reuse
the saved data. Cannery rechecks saved ranges before reuse.

Use `--restart` to remove only the saved state matching that identity and start
again:

```sh
uv run truss volume pull bdn://default/weights:local ./downloaded \
  --include model.safetensors \
  --restart
```

When a pull starts, Cannery cleans up eligible resume state in the destination
parent after 30 days without an update. Resume staging directories are
owner-only (`0700`), and checkpoint and lock files are owner-only (`0600`).
Checkpoints remain local but contain destination paths, selectors, and content
digests, so they must be treated as private. Durable pull resume currently
launches only on Unix systems; unsupported systems fail before transfer.

For a local server that enforces authentication, also set:

```sh
export TRUSS_CANNERY_AUTH_TOKEN_FILE=/path/to/cannery-token
chmod 600 /path/to/cannery-token
```

Fresh installs never assume a loopback API and never execute `cannery` from
`PATH`. Local mode requires both explicit `TRUSS_CANNERY_API` and
`TRUSS_CANNERY_BIN`. Developers may instead set
`TRUSS_CANNERY_ALLOW_PATH=1` to opt in to PATH lookup for that explicit
loopback endpoint. Remote use selects an exact platform artifact whose URL,
size, protocol version, and SHA-256 are trusted from the installed Truss
package. Downloads follow redirects only while the final URL remains HTTPS and
use Requests defaults, including `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY`, and
the normal corporate CA bundle mechanisms. TLS verification cannot be
disabled by this integration.

Truss first runs the non-mutating `cannery protocol` bootstrap and requires
machine protocol `1` with the `protobuf-delimited` encoding. The operation then
runs as
`cannery --machine-protocol 1 --machine-encoding protobuf-delimited ...`.
Stdout is a binary stream of canonical varint-length-delimited
`MachineRecordV1` messages; Truss reads and validates one bounded frame at a
time without decoding stdout as UTF-8. Typed progress, status, and warnings
render on stderr, and the typed terminal result becomes Truss text or
`--output json`. Cannery stderr is never parsed as protocol data. It is decoded
safely, drained concurrently, bounded, redacted, and retained only in failure
diagnostics.

To inspect a captured production stream, run
`cannery protocol decode [FILE|-]`. The decoder validates the same framing and
stream invariants and emits canonical ProtoJSON NDJSON for review. The checked
ProtoJSON fixtures are decoded debug artifacts; the
`fixtures/protobuf-delimited/*.bin` files are the authoritative production
goldens. Truss never falls back to ProtoJSON when binary negotiation fails.

The generated schema, Python types, protocol documentation, decoded ProtoJSON
fixtures, and authoritative binary goldens are vendored under
`truss/cli/cannery/generated`. Run
`scripts/sync_cannery_protocol_v1.py --check` to verify generated-code and hash
drift. Maintainers can pass `--source-root` pointing at a canonical contract
directory they supply to compare or synchronize the vendored files.

The check runs in unit tests, pre-commit, and PR CI. It validates the vendored
schema, generated Python files, documentation, fixtures, and hash manifest
against one another. For a protocol update, obtain the canonical protocol
source supplied by maintainers, pass its contract directory as `--source-root`,
commit all generated changes, and run `--check` locally.

Truss follows `next_page_token` from the Cannery namespace, reference, and
inspect metadata APIs until the final page. `volume ls` and `volume show`
therefore return complete collections by default; `--page-size` controls each
bounded request, not the total result size. Repeated, malformed, or excessive
tokens fail as protocol errors.

## Updating released artifacts

A trusted producer release manifest provides one `*.truss-pin.json` per
platform with exactly `cannery_version`, `protocol_version`,
`operating_system`, `architecture`, `url`, `size_bytes`, and `sha256`. After an
immutable public artifact release is published and independently verified,
perform a maintainer-reviewed import:

```sh
uv run python scripts/update_cannery_artifact_pins.py \
  /path/to/*.truss-pin.json
uv run python scripts/update_cannery_artifact_pins.py --check
```

The importer rejects extra manifest fields, unsupported platforms, mixed
versions, duplicate platforms, non-HTTPS URLs, and invalid sizes or digests.
The checked-in trust table remains empty until real artifacts are published;
do not copy unpublished build output into it.

Prototype binaries that implement only the previous split stdout/stderr
contract remain available for loopback development by explicitly setting
`TRUSS_CANNERY_PHASE0=1`. Truss never automatically downgrades, and the fallback
is rejected for non-loopback endpoints and pinned production artifacts.

Truss creates a correlation ID before authentication or network access.
Failures retain an owner-only, redacted local diagnostic log and print its
path; successful commands remove the temporary log. Truss never uploads
diagnostics automatically.
