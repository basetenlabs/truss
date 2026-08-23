from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Optional, Protocol, Tuple

import requests
import rich_click as click

_DOWNLOAD_TIMEOUT_SEC = (10, 120)
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class ArtifactMetadata:
    cannery_version: str
    protocol_version: int
    operating_system: str
    architecture: str
    url: str
    size_bytes: int
    sha256: str

    @property
    def platform_key(self) -> Tuple[str, str]:
        return (self.operating_system, self.architecture)


class StreamingResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def __enter__(self) -> "StreamingResponse": ...

    def __exit__(self, *_args: object) -> None: ...


class ArtifactHttpClient(Protocol):
    def get(self, url: str, **kwargs: object) -> StreamingResponse: ...


# No customer-native artifact has been published yet. Release automation adds
# reviewed, exact entries here; an empty trust anchor fails closed for remote use.
BUNDLED_ARTIFACTS: Mapping[Tuple[str, str], ArtifactMetadata] = MappingProxyType({})


def current_platform() -> Tuple[str, str]:
    operating_system = {"darwin": "macos", "linux": "linux"}.get(
        platform.system().lower()
    )
    architecture = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(platform.machine().lower())
    if operating_system is None or architecture is None:
        raise click.ClickException(
            "Cannery does not provide a native artifact for this platform."
        )
    return (operating_system, architecture)


def resolve_cannery_binary(
    *,
    allow_path_fallback: bool = False,
    artifacts: Optional[Mapping[Tuple[str, str], ArtifactMetadata]] = None,
    cache_dir: Optional[Path] = None,
    http_client: ArtifactHttpClient = requests,
) -> str:
    configured = os.environ.get("TRUSS_CANNERY_BIN")
    if configured:
        resolved = shutil.which(configured)
        if resolved is not None:
            return resolved
        raise click.ClickException(
            "Cannot execute the Cannery binary configured by TRUSS_CANNERY_BIN. "
            "Set it to an executable Cannery client."
        )

    platform_key = current_platform()
    trusted_artifacts = artifacts if artifacts is not None else BUNDLED_ARTIFACTS
    artifact = trusted_artifacts.get(platform_key)
    if artifact is not None:
        _validate_metadata(artifact, platform_key)
        return str(
            resolve_artifact(artifact, cache_dir=cache_dir, http_client=http_client)
        )

    if allow_path_fallback:
        resolved = shutil.which("cannery")
        if resolved is not None:
            return resolved
        raise click.ClickException(
            "Cannery client binary was not found on PATH. Install or build "
            "`cannery`, or set TRUSS_CANNERY_BIN to its executable path."
        )

    raise click.ClickException(
        "This Truss release has no pinned Cannery artifact for "
        f"{platform_key[0]}-{platform_key[1]}. Set TRUSS_CANNERY_BIN only for "
        "development or offline testing."
    )


def resolve_artifact(
    artifact: ArtifactMetadata,
    *,
    cache_dir: Optional[Path] = None,
    http_client: ArtifactHttpClient = requests,
) -> Path:
    _validate_metadata(artifact, artifact.platform_key)
    cache = cache_dir or _default_cache_dir()
    _prepare_private_cache(cache)
    destination = cache / f"cannery-{artifact.sha256.lower()}"
    if destination.exists() or destination.is_symlink():
        _verify_executable(destination, artifact)
        return destination

    _download_artifact(artifact, destination, http_client)
    _verify_executable(destination, artifact)
    return destination


def binary_diagnostic_metadata(
    path: Path, artifacts: Optional[Mapping[Tuple[str, str], ArtifactMetadata]] = None
) -> Tuple[str, str]:
    trusted_artifacts = artifacts if artifacts is not None else BUNDLED_ARTIFACTS
    for artifact in trusted_artifacts.values():
        if path.name == f"cannery-{artifact.sha256.lower()}":
            return (artifact.cannery_version, artifact.sha256.lower())
    try:
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            return ("development-override", "unavailable")
        return ("development-override", _sha256_file(path))
    except OSError:
        return ("development-override", "unavailable")


def _default_cache_dir() -> Path:
    configured = os.environ.get("TRUSS_CANNERY_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "truss" / "cannery"


def _validate_metadata(
    artifact: ArtifactMetadata, expected_platform: Tuple[str, str]
) -> None:
    if artifact.platform_key != expected_platform:
        raise click.ClickException(
            "Pinned Cannery artifact metadata does not match the selected platform."
        )
    if artifact.protocol_version != 1:
        raise click.ClickException(
            "Pinned Cannery artifact does not support the required protocol."
        )
    if artifact.size_bytes <= 0:
        raise click.ClickException("Pinned Cannery artifact has an invalid size.")
    digest = artifact.sha256.lower()
    if len(digest) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise click.ClickException(
            "Pinned Cannery artifact has an invalid SHA-256 digest."
        )
    if not artifact.url.startswith("https://"):
        raise click.ClickException("Pinned Cannery artifact URL must use HTTPS.")


def _prepare_private_cache(cache: Path) -> None:
    try:
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise click.ClickException(
            "Could not create the private Cannery binary cache."
        ) from None
    cache_stat = cache.lstat()
    if stat.S_ISLNK(cache_stat.st_mode) or not stat.S_ISDIR(cache_stat.st_mode):
        raise click.ClickException(
            "Cannery binary cache must be a regular directory, not a symlink."
        )
    if os.name != "nt" and cache_stat.st_uid != os.getuid():
        raise click.ClickException(
            "Cannery binary cache must be owned by the current user."
        )
    if os.name != "nt" and cache_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise click.ClickException(
            "Cannery binary cache must be owner-only (mode 0700)."
        )


def _download_artifact(
    artifact: ArtifactMetadata, destination: Path, http_client: ArtifactHttpClient
) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".cannery-download-", dir=destination.parent
    )
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            with http_client.get(
                artifact.url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    size_bytes += len(chunk)
                    if size_bytes > artifact.size_bytes:
                        raise click.ClickException(
                            "Downloaded Cannery artifact exceeds its pinned size."
                        )
                    digest.update(chunk)
                    file.write(chunk)
            file.flush()
            os.fsync(file.fileno())

        if size_bytes != artifact.size_bytes:
            raise click.ClickException(
                "Downloaded Cannery artifact size does not match its pin."
            )
        if digest.hexdigest() != artifact.sha256.lower():
            raise click.ClickException(
                "Downloaded Cannery artifact SHA-256 does not match its pin."
            )

        temp_path.chmod(0o700)
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        if isinstance(exc, (click.ClickException, KeyboardInterrupt, SystemExit)):
            raise
        raise click.ClickException(
            "Failed to download the pinned Cannery artifact over HTTPS."
        ) from None


def _verify_executable(path: Path, artifact: ArtifactMetadata) -> None:
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise click.ClickException(
            "Cached Cannery artifact must be a regular file, not a symlink."
        )
    if os.name != "nt" and path_stat.st_uid != os.getuid():
        raise click.ClickException(
            "Cached Cannery artifact must be owned by the current user."
        )
    if os.name != "nt" and path_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise click.ClickException(
            "Cached Cannery artifact must not be group- or world-writable."
        )
    if path_stat.st_nlink != 1:
        raise click.ClickException(
            "Cached Cannery artifact must not have additional hard links."
        )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise click.ClickException(
                "Cached Cannery artifact changed type during verification."
            )
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise click.ClickException(
                "Cached Cannery artifact changed during verification."
            )
        with os.fdopen(descriptor, "rb") as file:
            descriptor = -1
            digest = _sha256_stream(file)
        if opened_stat.st_size != artifact.size_bytes:
            raise click.ClickException(
                "Cached Cannery artifact size does not match its pin."
            )
        if digest != artifact.sha256.lower():
            raise click.ClickException(
                "Cached Cannery artifact SHA-256 does not match its pin."
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as file:
        return _sha256_stream(file)


def _sha256_stream(file) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: file.read(_DOWNLOAD_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
