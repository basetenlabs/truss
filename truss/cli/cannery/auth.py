from __future__ import annotations

import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
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
_AUTH_DIRECTORY_PREFIX = "truss-cannery-auth-"
_AUTH_DIRECTORY_PATTERN = re.compile(r"^truss-cannery-auth-(\d+)-")
_LEGACY_AUTH_DIRECTORY_MAX_AGE_SEC = 24 * 60 * 60
_MAX_REFRESH_WAIT_SEC = 5 * 60
_MIN_REFRESH_WAIT_SEC = 0.1
_REFRESH_LEEWAY_SEC = 60


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
    def refresh(self, token_file: Path) -> ExchangedToken: ...


@dataclass
class CanneryCredential:
    token_file: Optional[Path]
    mechanism: str
    _refresh_hook: Optional[TokenRefreshHook] = None
    _expires_at_epoch_sec: Optional[float] = None
    _cleanup_directory: Optional[Path] = None
    _refresh_stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _refresh_thread: Optional[threading.Thread] = field(
        default=None, init=False, repr=False
    )
    _refresh_error: Optional[CanneryClickException] = field(
        default=None, init=False, repr=False
    )

    def refresh(self) -> None:
        if self._refresh_hook is None or self.token_file is None:
            raise CanneryClickException(
                "The selected Cannery authentication mechanism does not support "
                "refresh."
            )
        self._refresh_hook.refresh(self.token_file)

    def close(self) -> None:
        self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join()
            self._refresh_thread = None
        if self._cleanup_directory is not None:
            shutil.rmtree(self._cleanup_directory)
            self._cleanup_directory = None

    def __enter__(self) -> "CanneryCredential":
        if (
            self._refresh_hook is not None
            and self.token_file is not None
            and self._expires_at_epoch_sec is not None
        ):
            self._refresh_thread = threading.Thread(
                target=self._refresh_until_stopped,
                name="truss-cannery-token-refresh",
                daemon=True,
            )
            self._refresh_thread.start()
        return self

    def __exit__(self, exception_type: object, *_args: object) -> None:
        self.close()
        if exception_type is None:
            self.raise_refresh_error()

    def raise_refresh_error(self) -> None:
        if self._refresh_error is not None:
            raise self._refresh_error

    def _refresh_until_stopped(self) -> None:
        assert self._refresh_hook is not None
        assert self.token_file is not None
        expires_at_epoch_sec = self._expires_at_epoch_sec
        while expires_at_epoch_sec is not None:
            wait_sec = _refresh_wait_sec(expires_at_epoch_sec)
            if self._refresh_stop.wait(wait_sec):
                return
            try:
                token = self._refresh_hook.refresh(self.token_file)
            except CanneryClickException as exc:
                self._refresh_error = exc
                return
            except Exception:
                self._refresh_error = CanneryClickException(
                    "Cannery credential refresh failed. No credential details "
                    "were logged."
                )
                return
            expires_at_epoch_sec = token.expires_at_epoch_sec


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

    def refresh(self, token_file: Path) -> ExchangedToken:
        token = _exchange_token(
            self._adapter, self._active_remote, self._correlation_id
        )
        _atomic_write_token(token_file, token.value)
        return token


class BasetenExchangeAuthProvider:
    def __init__(
        self, adapter: BasetenExchangeAdapter, active_remote: ActiveRemote
    ) -> None:
        self._adapter = adapter
        self._active_remote = active_remote

    def acquire(self, correlation_id: str) -> CanneryCredential:
        token = _exchange_token(self._adapter, self._active_remote, correlation_id)
        _cleanup_stale_auth_directories()
        token_directory = Path(
            tempfile.mkdtemp(prefix=f"{_AUTH_DIRECTORY_PREFIX}{os.getpid()}-")
        )
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
            _expires_at_epoch_sec=token.expires_at_epoch_sec,
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
    if token.expires_at_epoch_sec is not None and (
        isinstance(token.expires_at_epoch_sec, bool)
        or not isinstance(token.expires_at_epoch_sec, (int, float))
        or not math.isfinite(token.expires_at_epoch_sec)
    ):
        raise CanneryClickException(
            "Cannery credential exchange returned an invalid expiry."
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


def _refresh_wait_sec(expires_at_epoch_sec: float) -> float:
    remaining_sec = expires_at_epoch_sec - time.time()
    leeway_sec = min(_REFRESH_LEEWAY_SEC, max(0.0, remaining_sec / 5))
    return min(
        _MAX_REFRESH_WAIT_SEC, max(_MIN_REFRESH_WAIT_SEC, remaining_sec - leeway_sec)
    )


def _cleanup_stale_auth_directories() -> None:
    temp_root = Path(tempfile.gettempdir())
    try:
        candidates = list(temp_root.glob(f"{_AUTH_DIRECTORY_PREFIX}*"))
    except OSError:
        return
    for candidate in candidates:
        try:
            candidate_stat = candidate.lstat()
            if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(
                candidate_stat.st_mode
            ):
                continue
            if os.name != "nt" and candidate_stat.st_uid != os.getuid():
                continue
            match = _AUTH_DIRECTORY_PATTERN.match(candidate.name)
            if match is not None:
                process_id = int(match.group(1))
                if process_id == os.getpid() or _process_is_running(process_id):
                    continue
            elif time.time() - candidate_stat.st_mtime < (
                _LEGACY_AUTH_DIRECTORY_MAX_AGE_SEC
            ):
                continue
            shutil.rmtree(candidate)
        except OSError:
            continue


def _process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


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
