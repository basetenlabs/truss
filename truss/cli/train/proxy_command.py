#!/usr/bin/env python3
"""SSH helper for Baseten workloads (training jobs and inference models).

This script is installed to ~/.ssh/baseten/proxy-command.py by `truss train ssh setup`.
It is fully self-contained (only Python stdlib imports, no truss dependency).

It has two modes, both invoked by SSH automatically:

  1. Sign mode (Match exec): Signs an SSH certificate and caches the JWT.
     Invoked as: proxy-command.py --sign <hostname>

  2. Proxy mode (ProxyCommand): Connects to the SSH proxy using the cached JWT.
     Invoked as: proxy-command.py <hostname>

Hostname formats:
  training-job-<job_id>-<node>.<remote>.ssh.baseten.co        (training)
  model-<model_id>.<remote>.ssh.baseten.co                    (inference, default env)
  model-<model_id>-<env_name>.<remote>.ssh.baseten.co         (inference, specific env)
"""

import configparser
import dataclasses
import json
import os
import selectors
import socket
import ssl
import struct
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Stamped by `truss train ssh setup` with the truss version at install time.
# Sent to the signing API so the server can reject outdated clients.
CLIENT_VERSION = "{{CLIENT_VERSION}}"

BASETEN_SSH_DIR = Path(
    os.environ.get("BASETEN_SSH_DIR", Path.home() / ".ssh" / "baseten")
)
TRUSSRC_PATH = Path(os.environ.get("USER_TRUSSRC_PATH", Path.home() / ".trussrc"))
JWT_CACHE_DIR = BASETEN_SSH_DIR / ".jwt-cache"

HOSTNAME_SUFFIX = ".ssh.baseten.co"
HOSTNAME_PREFIX_TRAINING = "training-job-"
HOSTNAME_PREFIX_MODEL = "model-"

WORKLOAD_TRAINING = "training"
WORKLOAD_MODEL = "model"

BASETEN_REST_API_URL = os.environ.get("BASETEN_BASE_URL", "https://api.baseten.co")

STATUS_OK = 0x00


def error(msg):
    print(f"baseten-ssh: {msg}", file=sys.stderr)
    sys.exit(1)


def find_key_path():
    """Find the SSH private key (ed25519 or RSA fallback)."""
    for name in ("id_ed25519", "id_rsa"):
        path = BASETEN_SSH_DIR / name
        if path.exists():
            return path
    return None


@dataclasses.dataclass
class ParsedHostname:
    """Result of parsing an SSH hostname."""

    workload_type: str
    id: str
    replica: Optional[str]
    environment: Optional[str]
    remote: Optional[str]
    api_prefix: Optional[str]


def parse_hostname(hostname):
    """Parse hostname into a ParsedHostname.

    Supported formats:
      training-job-<job_id>-<node>[.<remote>[.<api_prefix>]].ssh.baseten.co
      model-<model_id>[.<remote>[.<api_prefix>]].ssh.baseten.co
      model-<model_id>-<env_name>[.<remote>[.<api_prefix>]].ssh.baseten.co
    """
    if not hostname.endswith(HOSTNAME_SUFFIX):
        error(f"Invalid hostname: {hostname} (expected *.ssh.baseten.co)")

    prefix = hostname[: -len(HOSTNAME_SUFFIX)]
    dot_idx = prefix.find(".")
    if dot_idx == -1:
        workload_part = prefix
        remote = None
        api_prefix = None
    else:
        workload_part = prefix[:dot_idx]
        rest = prefix[dot_idx + 1 :]
        parts = rest.split(".", 1)
        remote = parts[0]
        api_prefix = parts[1] if len(parts) > 1 else None

    if workload_part.startswith(HOSTNAME_PREFIX_TRAINING):
        return _parse_training_hostname(hostname, workload_part, remote, api_prefix)
    elif workload_part.startswith(HOSTNAME_PREFIX_MODEL):
        return _parse_model_hostname(hostname, workload_part, remote, api_prefix)
    else:
        error(
            f"Invalid hostname: {hostname} "
            f"(expected training-job-<job_id>-<node> or model-<model_id>)"
        )


def _parse_training_hostname(hostname, workload_part, remote, api_prefix):
    remainder = workload_part[len(HOSTNAME_PREFIX_TRAINING) :]

    dash_idx = remainder.rfind("-")
    if dash_idx == -1:
        error(f"Invalid hostname: {hostname} (cannot parse job_id and node)")

    job_id = remainder[:dash_idx]
    replica_str = remainder[dash_idx + 1 :]

    if not replica_str:
        error(f"Invalid hostname: {hostname} (empty replica)")
    if not replica_str.isdigit():
        error(f"Invalid hostname: {hostname} (replica must be a number)")
    if not job_id:
        error(f"Invalid hostname: {hostname} (empty job_id)")

    return ParsedHostname(
        workload_type=WORKLOAD_TRAINING,
        id=job_id,
        replica=replica_str,
        environment=None,
        remote=remote,
        api_prefix=api_prefix,
    )


def _parse_model_hostname(hostname, workload_part, remote, api_prefix):
    remainder = workload_part[len(HOSTNAME_PREFIX_MODEL) :]
    if not remainder:
        error(f"Invalid hostname: {hostname} (empty model_id)")

    # Model IDs are [a-z0-9]+ (no hyphens). If there's a hyphen, everything
    # after the first one is the environment name.
    dash_idx = remainder.find("-")
    if dash_idx == -1:
        model_id = remainder
        environment = None
    else:
        model_id = remainder[:dash_idx]
        environment = remainder[dash_idx + 1 :]
        if not environment:
            error(f"Invalid hostname: {hostname} (empty environment name)")

    if not model_id:
        error(f"Invalid hostname: {hostname} (empty model_id)")

    # Replica comes from env var, not hostname
    replica = os.environ.get("BASETEN_REPLICA") or None

    return ParsedHostname(
        workload_type=WORKLOAD_MODEL,
        id=model_id,
        replica=replica,
        environment=environment,
        remote=remote,
        api_prefix=api_prefix,
    )


def _read_trussrc():
    """Read and return the parsed ~/.trussrc config."""
    if not TRUSSRC_PATH.exists():
        error("~/.trussrc not found. Run 'truss login' first.")

    config = configparser.ConfigParser()
    config.read(TRUSSRC_PATH)
    return config


def resolve_remote(remote, config):
    """Resolve the remote name. If None, default to the only remote in ~/.trussrc."""
    if remote is None:
        sections = config.sections()
        if len(sections) == 1:
            remote = sections[0]
        elif len(sections) == 0:
            error("No remotes configured in ~/.trussrc. Run 'truss login' first.")
        else:
            error(
                f"Multiple remotes in ~/.trussrc: {', '.join(sections)}. "
                f"Specify one in the hostname: ssh training-job-<job_id>-<node>.<remote>.ssh.baseten.co"
            )
    return remote


def load_trussrc(remote, config=None):
    """Read ~/.trussrc and return (api_key, remote_url) for the given remote."""
    if config is None:
        config = _read_trussrc()

    if not config.has_section(remote):
        error(
            f"Remote '{remote}' not found in ~/.trussrc. Available: {', '.join(config.sections())}"
        )

    api_key = config.get(remote, "api_key", fallback=None)
    remote_url = config.get(remote, "remote_url", fallback=None)

    if not api_key:
        error(f"No api_key for remote '{remote}' in ~/.trussrc")
    if not remote_url:
        error(f"No remote_url for remote '{remote}' in ~/.trussrc")

    return api_key, remote_url


def resolve_rest_api_url(api_prefix=None):
    """Determine the REST API URL.

    If api_prefix is set (from hostname), use https://api.<prefix>.baseten.co.
    Otherwise use the default production URL.
    """
    if api_prefix:
        return f"https://api.{api_prefix}.baseten.co"
    return BASETEN_REST_API_URL


def api_request(url, api_key, method="GET", body=None, extra_headers=None):
    """Make an API request, return parsed JSON or raise on error."""
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    headers["X-Client-Version"] = CLIENT_VERSION

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
            # API errors can come in different formats
            if isinstance(err_body, dict):
                msg = (
                    err_body.get("error")
                    or err_body.get("detail")
                    or err_body.get("message")
                    or str(err_body)
                )
            elif isinstance(err_body, list) and err_body:
                msg = err_body[0] if isinstance(err_body[0], str) else str(err_body[0])
            else:
                msg = str(err_body)
        except Exception:
            msg = e.reason
        error(f"API error ({e.code}): {msg}")
    except urllib.error.URLError as e:
        error(f"Cannot reach API: {e.reason}")


def resolve_project_id(rest_url, api_key, job_id):
    """Look up project_id for a job via the search API."""
    resp = api_request(
        f"{rest_url}/v1/training_jobs/search",
        api_key,
        method="POST",
        body={"job_id": job_id},
    )

    jobs = resp if isinstance(resp, list) else resp.get("training_jobs", [])
    if not jobs:
        error(f"Job '{job_id}' not found. Is the job ID correct?")

    return jobs[0]["training_project"]["id"]


def sign_training_certificate(
    rest_url, api_key, project_id, job_id, public_key, replica, key_path
):
    """Call the training SSH signing endpoint. Returns (jwt, proxy_address)."""
    resp = api_request(
        f"{rest_url}/v1/training_projects/{project_id}/jobs/{job_id}/ssh/sign",
        api_key,
        method="POST",
        body={"public_key": public_key, "replica_id": replica},
    )

    cert_path = key_path.parent / (key_path.name + "-cert.pub")
    cert_path.write_text(resp["ssh_certificate"])
    return resp["jwt"], resp["proxy_address"]


def sign_model_certificate(rest_url, api_key, parsed, public_key, key_path):
    """Call the inference SSH signing endpoint. Returns (jwt, proxy_address)."""
    body = {"public_key": public_key}
    if parsed.replica:
        body["replica_id"] = parsed.replica

    if parsed.environment:
        url = f"{rest_url}/v1/models/{parsed.id}/environments/{parsed.environment}/ssh/sign"
    else:
        url = f"{rest_url}/v1/models/{parsed.id}/ssh/sign"

    resp = api_request(url, api_key, method="POST", body=body)

    cert_path = key_path.parent / (key_path.name + "-cert.pub")
    cert_path.write_text(resp["ssh_certificate"])
    return resp["jwt"], resp["proxy_address"]


# --- JWT cache: shared between sign mode and proxy mode ---


def _jwt_cache_path(parsed):
    if parsed.workload_type == WORKLOAD_TRAINING:
        name = f"{parsed.id}-{parsed.replica}"
    else:
        parts = [parsed.id]
        if parsed.environment:
            parts.append(parsed.environment)
        if parsed.replica:
            parts.append(parsed.replica)
        name = "-".join(parts)
    return JWT_CACHE_DIR / f"{name}.json"


def save_jwt_cache(parsed, jwt_token, proxy_address):
    """Save JWT and proxy address for the proxy command to read."""
    JWT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _jwt_cache_path(parsed)
    path.write_text(json.dumps({"jwt": jwt_token, "proxy_address": proxy_address}))


def load_jwt_cache(parsed):
    """Load cached JWT and proxy address. Returns (jwt, proxy_address) or None."""
    path = _jwt_cache_path(parsed)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data["jwt"], data["proxy_address"]
    except (json.JSONDecodeError, KeyError):
        return None


# --- Sign mode (Match exec) ---


def _sign_for_parsed(parsed, rest_url, api_key, key_path):
    """Sign a certificate for the given parsed hostname. Returns (jwt, proxy_address)."""
    public_key = key_path.with_suffix(".pub").read_text().strip()

    if parsed.workload_type == WORKLOAD_TRAINING:
        project_id = resolve_project_id(rest_url, api_key, parsed.id)
        return sign_training_certificate(
            rest_url,
            api_key,
            project_id,
            parsed.id,
            public_key,
            parsed.replica,
            key_path,
        )
    else:
        return sign_model_certificate(rest_url, api_key, parsed, public_key, key_path)


def main_sign():
    """Sign mode: called by Match exec before SSH loads identity files.

    Signs the certificate, writes it to disk, and caches the JWT for
    the ProxyCommand to use. Always exits 0 so the Match block applies.
    """
    if len(sys.argv) < 3:
        error("Usage: proxy-command.py --sign <hostname>")

    hostname = sys.argv[2]

    key_path = find_key_path()
    if not key_path:
        print(
            "baseten-ssh: SSH keypair not found. Run 'truss train ssh setup' first.",
            file=sys.stderr,
        )
        sys.exit(0)  # Exit 0 so Match still applies; ProxyCommand will show error too

    parsed = parse_hostname(hostname)
    config = _read_trussrc()
    remote = resolve_remote(parsed.remote, config)
    api_key, _ = load_trussrc(remote, config)
    rest_url = resolve_rest_api_url(parsed.api_prefix)

    jwt_token, proxy_address = _sign_for_parsed(parsed, rest_url, api_key, key_path)
    save_jwt_cache(parsed, jwt_token, proxy_address)


# --- Proxy mode (ProxyCommand) ---


def connect_proxy(proxy_address, jwt_token):
    """TLS connect to proxy, send JWT, return connected socket."""
    host, port_str = proxy_address.rsplit(":", 1)
    port = int(port_str)

    raw_sock = socket.create_connection((host, port), timeout=10)

    if os.environ.get("BASETEN_SSH_PROXY_INSECURE", "").lower() in ("1", "true"):
        tls_sock = raw_sock
    else:
        ctx = ssl.create_default_context()
        tls_sock = ctx.wrap_socket(raw_sock, server_hostname=host)

    # Send length-prefixed JWT
    jwt_bytes = jwt_token.encode()
    tls_sock.sendall(struct.pack("!I", len(jwt_bytes)))
    tls_sock.sendall(jwt_bytes)

    # Read status byte
    status = tls_sock.recv(1)
    if not status or status[0] != STATUS_OK:
        tls_sock.close()
        error(
            "SSH proxy rejected the connection. Is the workload and SSH server running?"
        )

    return tls_sock


def relay_selectors(tls_sock):
    """Bidirectional relay using selectors (macOS/Linux)."""
    stdin_fd = sys.stdin.buffer.fileno()
    stdout_fd = sys.stdout.buffer.fileno()

    sel = selectors.DefaultSelector()
    sel.register(stdin_fd, selectors.EVENT_READ, "stdin")
    sel.register(tls_sock, selectors.EVENT_READ, "socket")

    try:
        while True:
            events = sel.select()
            for key, _ in events:
                if key.data == "stdin":
                    data = os.read(stdin_fd, 65536)
                    if not data:
                        return
                    tls_sock.sendall(data)
                elif key.data == "socket":
                    data = tls_sock.recv(65536)
                    if not data:
                        return
                    os.write(stdout_fd, data)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        sel.close()


def relay_threads(tls_sock):
    """Bidirectional relay using threads (Windows)."""
    done = threading.Event()

    def stdin_to_socket():
        try:
            while not done.is_set():
                data = os.read(sys.stdin.buffer.fileno(), 65536)
                if not data:
                    break
                tls_sock.sendall(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            done.set()

    def socket_to_stdout():
        try:
            while not done.is_set():
                data = tls_sock.recv(65536)
                if not data:
                    break
                os.write(sys.stdout.buffer.fileno(), data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            done.set()

    t1 = threading.Thread(target=stdin_to_socket, daemon=True)
    t2 = threading.Thread(target=socket_to_stdout, daemon=True)
    t1.start()
    t2.start()
    done.wait()


def main_proxy():
    """Proxy mode: called by ProxyCommand after sign mode has run."""
    if len(sys.argv) < 2:
        error("Usage: proxy-command.py <hostname>")

    hostname = sys.argv[1]

    key_path = find_key_path()
    if not key_path:
        error("SSH keypair not found. Run 'truss train ssh setup' first.")

    parsed = parse_hostname(hostname)

    # Try to use cached JWT from sign mode
    cached = load_jwt_cache(parsed)
    if cached:
        jwt_token, proxy_address = cached
    else:
        # Fallback: sign here if Match exec didn't run (e.g. old SSH config)
        config = _read_trussrc()
        remote = resolve_remote(parsed.remote, config)
        api_key, _ = load_trussrc(remote, config)
        rest_url = resolve_rest_api_url(parsed.api_prefix)
        jwt_token, proxy_address = _sign_for_parsed(parsed, rest_url, api_key, key_path)

    # Connect to proxy
    tls_sock = connect_proxy(proxy_address, jwt_token)

    # Relay stdin/stdout
    try:
        if sys.platform == "win32":
            relay_threads(tls_sock)
        else:
            relay_selectors(tls_sock)
    finally:
        tls_sock.close()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--sign":
        main_sign()
    else:
        main_proxy()
