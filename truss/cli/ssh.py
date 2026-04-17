"""SSH setup logic for `truss train ssh` commands."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import truss

BASETEN_SSH_DIR = Path(
    os.environ.get("BASETEN_SSH_DIR", Path.home() / ".ssh" / "baseten")
)
SSH_CONFIG_PATH = Path(
    os.environ.get("BASETEN_SSH_CONFIG_PATH", Path.home() / ".ssh" / "config")
)

MARKER_START = "# --- baseten-ssh ---"
MARKER_END = "# --- end baseten-ssh ---"

SSH_CONFIG_BLOCK_UNIX = """\
{marker_start}
Match host training-job-*.ssh.baseten.co exec "{python} {proxy_script} --sign %n"
    ProxyCommand sh -c 'test -x "{python}" || {{ echo "baseten-ssh: Python not found at {python}. Please try re-running: truss ssh setup" >&2; exit 127; }}; exec "{python}" "{proxy_script}" "$1"' -- %n
    User baseten
    IdentityFile {key_path}
    CertificateFile {cert_path}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Match host model-*.ssh.baseten.co exec "{python} {proxy_script} --sign %n"
    ProxyCommand sh -c 'test -x "{python}" || {{ echo "baseten-ssh: Python not found at {python}. Please try re-running: truss ssh setup" >&2; exit 127; }}; exec "{python}" "{proxy_script}" "$1"' -- %n
    User app
    IdentityFile {key_path}
    CertificateFile {cert_path}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
{marker_end}
"""

SSH_CONFIG_BLOCK_WINDOWS = """\
{marker_start}
Match host training-job-*.ssh.baseten.co exec "\\"{python}\\" \\"{proxy_script}\\" --sign %n"
    ProxyCommand "{python}" "{proxy_script}" %n
    User baseten
    IdentityFile {key_path}
    CertificateFile {cert_path}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Match host model-*.ssh.baseten.co exec "\\"{python}\\" \\"{proxy_script}\\" --sign %n"
    ProxyCommand "{python}" "{proxy_script}" %n
    User app
    IdentityFile {key_path}
    CertificateFile {cert_path}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
{marker_end}
"""

PROXY_COMMAND_SOURCE = Path(__file__).parent / "proxy_command.py"


def _resolve_python() -> str:
    """Find a stable Python 3.10+ interpreter for the ProxyCommand.

    System python3 on macOS can be as old as 3.9 with broken TLS 1.3
    support. We prefer a system-wide install over a venv Python since
    venvs can be deleted/recreated, breaking the hardcoded path.
    """
    min_version = (3, 10)

    def _is_venv(path: str) -> bool:
        return "/.venv/" in path or "/venv/" in path

    def _check_version(path: str) -> bool:
        try:
            result = subprocess.run(
                [
                    path,
                    "-c",
                    "import sys; print(sys.version_info.major, sys.version_info.minor)",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                major, minor = result.stdout.strip().split()
                return (int(major), int(minor)) >= min_version
        except Exception:
            pass
        return False

    # First: look for a system-wide Python 3.10+ (survives venv changes)
    # Include "python" for Windows where "python3" doesn't exist,
    # and "py" for the Windows Python Launcher.
    candidates = [
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3",
        "python",
    ]
    if sys.platform == "win32":
        candidates.append("py")

    for name in candidates:
        path = shutil.which(name)
        if path and not _is_venv(path) and _check_version(path):
            return path

    raise RuntimeError(
        "Could not find Python 3.10+ on your system. "
        "Re-run with an explicit path: truss train ssh setup --python /path/to/python3"
    )


def ensure_ssh_keypair(key_dir: Path = BASETEN_SSH_DIR) -> tuple[Path, bool]:
    """Generate SSH keypair if it doesn't exist.

    Returns (private_key_path, reused) where reused is True if an existing key was found.
    Prefers ed25519, falls back to RSA if ssh-keygen doesn't support it.
    """
    key_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(key_dir, 0o700)

    for name in ("id_ed25519", "id_rsa"):
        if (key_dir / name).exists():
            return key_dir / name, True

    key_path = key_dir / "id_ed25519"
    result = subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-f",
            str(key_path),
            "-N",
            "",
            "-C",
            "baseten-training",
            "-q",
        ],
        capture_output=True,
    )
    if result.returncode == 0:
        return key_path, False

    key_path = key_dir / "id_rsa"
    subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "rsa",
            "-b",
            "4096",
            "-f",
            str(key_path),
            "-N",
            "",
            "-C",
            "baseten-training",
            "-q",
        ],
        check=True,
    )
    return key_path, False


def install_proxy_command_script(
    key_dir: Path = BASETEN_SSH_DIR, default_remote: Optional[str] = None
) -> Path:
    """Install the proxy-command.py script to key_dir with the current truss version stamped in."""
    key_dir.mkdir(parents=True, exist_ok=True)

    source_content = PROXY_COMMAND_SOURCE.read_text()
    stamped_content = source_content.replace("{{CLIENT_VERSION}}", truss.__version__)
    stamped_content = stamped_content.replace(
        "{{DEFAULT_REMOTE}}", default_remote or ""
    )

    dest = key_dir / "proxy-command.py"
    dest.write_text(stamped_content)
    os.chmod(dest, 0o700)
    return dest


def setup_ssh_config(
    key_dir: Path = BASETEN_SSH_DIR,
    key_path: Optional[Path] = None,
    python_override: Optional[str] = None,
) -> None:
    """Add or replace the baseten-ssh block in ~/.ssh/config."""
    SSH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    proxy_script = key_dir / "proxy-command.py"
    if key_path is None:
        key_path = key_dir / "id_ed25519"
    cert_path = key_path.parent / (key_path.name + "-cert.pub")

    template = (
        SSH_CONFIG_BLOCK_WINDOWS if sys.platform == "win32" else SSH_CONFIG_BLOCK_UNIX
    )
    block = template.format(
        marker_start=MARKER_START,
        marker_end=MARKER_END,
        python=python_override or _resolve_python(),
        proxy_script=proxy_script,
        key_path=key_path,
        cert_path=cert_path,
    )

    if SSH_CONFIG_PATH.exists():
        existing = SSH_CONFIG_PATH.read_text()
    else:
        existing = ""

    # Replace existing block or append
    start_idx = existing.find(MARKER_START)
    end_idx = existing.find(MARKER_END)

    if start_idx != -1 and end_idx != -1:
        after_idx = end_idx + len(MARKER_END)
        # Consume a single trailing newline if present (\r\n or \n), so the
        # replacement block (which ends with \n) doesn't double the separator.
        # Never consume unconditionally — that would eat the first char of the
        # next config entry when MARKER_END sits at EOF with no newline.
        if existing[after_idx : after_idx + 2] == "\r\n":
            after_idx += 2
        elif existing[after_idx : after_idx + 1] == "\n":
            after_idx += 1
        new_content = existing[:start_idx] + block + existing[after_idx:]
    else:
        separator = "\n" if existing and not existing.endswith("\n") else ""
        new_content = existing + separator + block

    SSH_CONFIG_PATH.write_text(new_content)
    os.chmod(SSH_CONFIG_PATH, 0o644)


def is_setup_complete(key_dir: Path = BASETEN_SSH_DIR) -> bool:
    """Check if SSH setup has been completed."""
    if not (key_dir / "proxy-command.py").exists():
        return False
    return any((key_dir / name).exists() for name in ("id_ed25519", "id_rsa"))
