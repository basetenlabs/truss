# Cannery volume CLI MVP

`truss volume push|ls|show|pull` delegates volume operations to the Cannery
client binary. This is a local, developer-facing vertical slice for RUN-871;
Truss does not package the binary yet.

## Temporary configuration

The following environment variables bridge Truss to Cannery:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRUSS_CANNERY_BIN` | Cannery client executable | `cannery` on `PATH` |
| `TRUSS_CANNERY_API` | Cannery API endpoint | `http://127.0.0.1:8787` |
| `TRUSS_CANNERY_ORG` | Cannery organization | `dev` |
| `TRUSS_CANNERY_AUTH_TOKEN_FILE` | Existing Cannery bearer-token file | unset |

This authentication path is temporary pending RUN-869. Truss does not exchange
a Baseten API token for a Cannery token yet. An explicit token file is required
for every non-loopback endpoint. Only `localhost`, `127.0.0.0/8`, and `::1`
endpoints may run without a token, for local development. Truss passes an
explicit token file to the child as `CANNERY_AUTH_TOKEN_FILE`; this does not
change Cannery server authentication. Truss forwards the selected organization
as `CANNERY_ORG`; Cannery versions that no longer need a separate organization
selector ignore it.

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
```

The subprocess contract is `cannery -o json --progress machine`: one final,
protocol-versioned JSON object on stdout, plus protocol-versioned NDJSON
progress events on stderr.
Use `--output json` on a Truss volume subcommand to preserve the final object on
stdout for scripts; progress remains on stderr.
