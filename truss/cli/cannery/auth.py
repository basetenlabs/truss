from __future__ import annotations

import os
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol

from truss.cli.cannery.config import ActiveRemote, CanneryConfig
from truss.cli.cannery.errors import CanneryClickException, CanneryUsageError

_PARENT_CREDENTIAL_ENV_KEYS = (
    "BASETEN_API_KEY",
    "BASETEN_TRUSS_AUTH_API_KEY",
    "CANNERY_AUTH_TOKEN",
    "CANNERY_AUTH_TOKEN_FILE",
    "TRUSS_CANNERY_AUTH_TOKEN_FILE",
)


@dataclass(frozen=True)
class ExchangedToken:
    value: str
    expires_at_epoch_sec: Optional[float] = None


class BasetenExchangeAdapter(Protocol):
    """Adapter for production token exchange."""

    def exchange(
        self, active_remote: ActiveRemote, correlation_id: str
    ) -> ExchangedToken: ...


class TokenRefreshHook(Protocol):
    def refresh(self, token_file: Path) -> None: ...


@dataclass
class CanneryCredential:
    token_file: Optional[Path]
    mechanism: str
    _refresh_hook: Optional[TokenRefreshHook] = None
    _cleanup_directory: Optional[Path] = None

    def refresh(self) -> None:
        if self._refresh_hook is None or self.token_file is None:
            raise CanneryClickException(
                "The selected Cannery authentication mechanism does not support "
                "refresh."
            )
        self._refresh_hook.refresh(self.token_file)

    def close(self) -> None:
        if self._cleanup_directory is not None:
            shutil.rmtree(self._cleanup_directory)
            self._cleanup_directory = None

    def __enter__(self) -> "CanneryCredential":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class CanneryAuthProvider(Protocol):
    def acquire(self, correlation_id: str) -> CanneryCredential: ...


class LoopbackNoAuthProvider:
    def acquire(self, correlation_id: str) -> CanneryCredential:
        del correlation_id
        return CanneryCredential(token_file=None, mechanism="loopback-no-auth")


class ExplicitTokenFileAuthProvider:
    def __init__(self, token_file: Path) -> None:
        self._token_file = token_file.expanduser()

    def acquire(self, correlation_id: str) -> CanneryCredential:
        del correlation_id
        token_file = _validate_owner_only_file(self._token_file)
        return CanneryCredential(token_file=token_file, mechanism="explicit-token-file")


class _ExchangeRefreshHook:
    def __init__(
        self,
        adapter: BasetenExchangeAdapter,
        active_remote: ActiveRemote,
        correlation_id: str,
    ) -> None:
        self._adapter = adapter
        self._active_remote = active_remote
        self._correlation_id = correlation_id

    def refresh(self, token_file: Path) -> None:
        token = _exchange_token(
            self._adapter, self._active_remote, self._correlation_id
        )
        _atomic_write_token(token_file, token.value)


class BasetenExchangeAuthProvider:
    def __init__(
        self, adapter: BasetenExchangeAdapter, active_remote: ActiveRemote
    ) -> None:
        self._adapter = adapter
        self._active_remote = active_remote

    def acquire(self, correlation_id: str) -> CanneryCredential:
        token = _exchange_token(self._adapter, self._active_remote, correlation_id)
        token_directory = Path(tempfile.mkdtemp(prefix="truss-cannery-auth-"))
        try:
            token_directory.chmod(0o700)
            token_file = token_directory / "token"
            _atomic_write_token(token_file, token.value)
        except BaseException:
            shutil.rmtree(token_directory)
            raise
        return CanneryCredential(
            token_file=token_file,
            mechanism="baseten-exchange",
            _refresh_hook=_ExchangeRefreshHook(
                self._adapter, self._active_remote, correlation_id
            ),
            _cleanup_directory=token_directory,
        )


def _exchange_token(
    adapter: BasetenExchangeAdapter, active_remote: ActiveRemote, correlation_id: str
) -> ExchangedToken:
    try:
        token = adapter.exchange(active_remote, correlation_id)
    except Exception:
        raise CanneryClickException(
            "Cannery credential exchange failed. No credential details were logged."
        ) from None
    if not token.value:
        raise CanneryClickException(
            "Cannery credential exchange returned an empty token."
        )
    return token


def _atomic_write_token(token_file: Path, token: str) -> None:
    token_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_file = token_file.parent / f".token-{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(token)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_file, token_file)
        token_file.chmod(0o600)
        _fsync_directory(token_file.parent)
    except BaseException:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_owner_only_file(token_file: Path) -> Path:
    try:
        file_stat = token_file.lstat()
    except FileNotFoundError:
        raise CanneryUsageError(
            "TRUSS_CANNERY_AUTH_TOKEN_FILE must point to an existing token file."
        ) from None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise CanneryUsageError(
            "TRUSS_CANNERY_AUTH_TOKEN_FILE must be a regular file, not a symlink."
        )
    if os.name != "nt" and file_stat.st_uid != os.getuid():
        raise CanneryUsageError(
            "TRUSS_CANNERY_AUTH_TOKEN_FILE must be owned by the current user."
        )
    if os.name != "nt" and file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise CanneryUsageError(
            "TRUSS_CANNERY_AUTH_TOKEN_FILE must be owner-only (mode 0600)."
        )
    return token_file.resolve()


def select_auth_provider(
    config: CanneryConfig, exchange_adapter: Optional[BasetenExchangeAdapter]
) -> CanneryAuthProvider:
    explicit_token_file = os.environ.get("TRUSS_CANNERY_AUTH_TOKEN_FILE")
    if explicit_token_file:
        return ExplicitTokenFileAuthProvider(Path(explicit_token_file))
    if config.is_loopback:
        return LoopbackNoAuthProvider()
    if config.active_remote is None:
        raise CanneryUsageError(
            "A Cannery token is required for non-loopback endpoints. Set "
            "TRUSS_CANNERY_AUTH_TOKEN_FILE to an owner-only token file for "
            "development, or use a configured Truss remote that supports "
            "production token exchange."
        )
    if exchange_adapter is None:
        raise CanneryUsageError(
            "Production token exchange is not available in this Truss build. "
            "For development, set TRUSS_CANNERY_AUTH_TOKEN_FILE to an owner-only "
            "token file."
        )
    return BasetenExchangeAuthProvider(exchange_adapter, config.active_remote)


def child_environment(
    credential: CanneryCredential, org: str, correlation_id: str, diagnostic_path: Path
) -> Dict[str, str]:
    env = os.environ.copy()
    for key in _PARENT_CREDENTIAL_ENV_KEYS:
        env.pop(key, None)
    env["CANNERY_ORG"] = org
    env["CANNERY_CORRELATION_ID"] = correlation_id
    env["CANNERY_DIAGNOSTIC_LOG"] = str(diagnostic_path)
    if credential.token_file is not None:
        env["CANNERY_AUTH_TOKEN_FILE"] = str(credential.token_file)
    return env
