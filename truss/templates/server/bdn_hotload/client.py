from __future__ import annotations

import http.client
import json
import socket
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Union

DEFAULT_SOCKET_PATH = Path("/run/bdn/hotload.sock")
_MOUNT_COLLECTION_PATH = "/v1/hotload"


class MountState(str, Enum):
    ACCEPTED = "ACCEPTED"
    RESOLVING = "RESOLVING"
    MOUNTING = "MOUNTING"
    READY = "READY"
    FAILED = "FAILED"
    UNMOUNTING = "UNMOUNTING"


@dataclass(frozen=True)
class MountFailure:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True)
class Mount:
    id: str
    source: str
    target: str
    path: str
    pinned_source: Optional[str]
    state: MountState
    error: Optional[MountFailure]


class HotLoadError(Exception):
    """Base class for Hot Load client errors."""


class HotLoadConnectionError(HotLoadError):
    """The local Hot Load socket could not complete a request."""


class HotLoadProtocolError(HotLoadError):
    """The server returned a response that does not match the API contract."""


class HotLoadAPIError(HotLoadError):
    def __init__(
        self, status_code: int, code: str, message: str, retryable: bool
    ) -> None:
        super().__init__(
            f"Hot Load request failed with HTTP {status_code} ({code}): {message}"
        )
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable


class HotLoadMountError(HotLoadError):
    def __init__(self, mount: Mount) -> None:
        failure = mount.error or MountFailure(
            code="MOUNT_FAILED",
            message="mount failed without an error body",
            retryable=False,
        )
        super().__init__(
            f"Hot Load mount {mount.id} failed ({failure.code}): {failure.message}"
        )
        self.mount = mount
        self.code = failure.code
        self.message = failure.message
        self.retryable = failure.retryable


class HotLoadTimeoutError(HotLoadError):
    def __init__(self, mount_id: str, timeout_sec: float, last_mount: Mount) -> None:
        super().__init__(
            f"Hot Load mount {mount_id} did not become ready within "
            f"{timeout_sec:g} seconds; last state was {last_mount.state.value}"
        )
        self.mount_id = mount_id
        self.timeout_sec = timeout_sec
        self.last_mount = last_mount


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout_sec: float) -> None:
        super().__init__("localhost", timeout=timeout_sec)
        self._socket_path = socket_path

    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            connection.connect(str(self._socket_path))
        except BaseException:
            connection.close()
            raise
        self.sock = connection


class HotLoadClient:
    """Synchronous client for one pod's local Hot Load API."""

    def __init__(
        self,
        socket_path: Union[str, Path] = DEFAULT_SOCKET_PATH,
        *,
        request_timeout_sec: float = 10.0,
        poll_interval_sec: float = 0.1,
        max_retries: int = 2,
    ) -> None:
        if request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive")
        if poll_interval_sec < 0:
            raise ValueError("poll_interval_sec must not be negative")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.socket_path = Path(socket_path)
        self.request_timeout_sec = request_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.max_retries = max_retries

    def mount(
        self,
        source: str,
        target: str,
        *,
        wait: bool = True,
        idempotency_key: Optional[str] = None,
        wait_timeout_sec: float = 300.0,
    ) -> Mount:
        mount = self.create_mount(source, target, idempotency_key=idempotency_key)
        if not wait:
            return mount
        return self.wait_until_ready(
            mount.id, timeout_sec=wait_timeout_sec, initial_mount=mount
        )

    def create_mount(
        self, source: str, target: str, *, idempotency_key: Optional[str] = None
    ) -> Mount:
        key = idempotency_key or uuid.uuid4().hex
        payload = self._request(
            "POST",
            _MOUNT_COLLECTION_PATH,
            body={"source": source, "target": target},
            headers={"Idempotency-Key": key},
            retry_transport=True,
        )
        return _parse_mount(payload)

    def list_mounts(self) -> list[Mount]:
        payload = _expect_mapping(
            self._request("GET", _MOUNT_COLLECTION_PATH), "mount list"
        )
        raw_mounts = payload.get("mounts")
        if not isinstance(raw_mounts, list):
            raise HotLoadProtocolError(
                "mount list response must contain a mounts array"
            )
        return [_parse_mount(raw_mount) for raw_mount in raw_mounts]

    def get_mount(self, mount_id: str) -> Mount:
        return _parse_mount(self._request("GET", _mount_path(mount_id)))

    def delete_mount(self, mount_id: str) -> None:
        self._request("DELETE", _mount_path(mount_id))

    def wait_until_ready(
        self,
        mount_id: str,
        *,
        timeout_sec: float = 300.0,
        initial_mount: Optional[Mount] = None,
    ) -> Mount:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        deadline = time.monotonic() + timeout_sec
        mount = initial_mount or self.get_mount(mount_id)
        while True:
            if mount.state == MountState.READY:
                return mount
            if mount.state == MountState.FAILED:
                raise HotLoadMountError(mount)
            if mount.state == MountState.UNMOUNTING:
                raise HotLoadProtocolError(
                    f"Hot Load mount {mount_id} began unmounting before it became ready"
                )
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0:
                raise HotLoadTimeoutError(mount_id, timeout_sec, mount)
            time.sleep(min(self.poll_interval_sec, remaining_sec))
            mount = self.get_mount(mount_id)

    def unmount(self, mount_id: str, *, wait_timeout_sec: float = 300.0) -> None:
        mount = self.get_mount(mount_id)
        if mount.state in {
            MountState.ACCEPTED,
            MountState.RESOLVING,
            MountState.MOUNTING,
        }:
            try:
                self.wait_until_ready(
                    mount_id, timeout_sec=wait_timeout_sec, initial_mount=mount
                )
            except HotLoadMountError:
                pass
        self.delete_mount(mount_id)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        retry_transport: bool = False,
    ) -> Any:
        request_headers = {"Accept": "application/json", **(headers or {})}
        encoded_body = None
        if body is not None:
            encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        retry_request = retry_transport or method == "GET"
        attempts = self.max_retries + 1 if retry_request else 1
        for attempt in range(attempts):
            connection = _UnixHTTPConnection(self.socket_path, self.request_timeout_sec)
            try:
                connection.request(
                    method, path, body=encoded_body, headers=request_headers
                )
                response = connection.getresponse()
                response_body = response.read()
            except (OSError, http.client.HTTPException) as error:
                if attempt + 1 == attempts:
                    raise HotLoadConnectionError(
                        f"could not call Hot Load through {self.socket_path}: {error}"
                    ) from error
                time.sleep(self.poll_interval_sec)
                continue
            finally:
                connection.close()

            payload = _decode_json(response_body) if response_body else None
            if 200 <= response.status < 300:
                return payload
            api_error = _parse_api_error(response.status, payload)
            if not (retry_request and api_error.retryable and attempt + 1 < attempts):
                raise api_error
            time.sleep(self.poll_interval_sec)
        raise RuntimeError("unreachable Hot Load request retry state")


def _mount_path(mount_id: str) -> str:
    if not mount_id or "/" in mount_id:
        raise ValueError("mount_id must be a non-empty path segment")
    return f"{_MOUNT_COLLECTION_PATH}/{mount_id}"


def _decode_json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HotLoadProtocolError("Hot Load returned invalid JSON") from error


def _parse_api_error(status_code: int, payload: Any) -> HotLoadAPIError:
    envelope = _expect_mapping(payload, "error response")
    error = _expect_mapping(envelope.get("error"), "error response body")
    return HotLoadAPIError(
        status_code=status_code,
        code=_expect_str(error, "code"),
        message=_expect_str(error, "message"),
        retryable=_expect_bool(error, "retryable"),
    )


def _parse_mount(payload: Any) -> Mount:
    raw = _expect_mapping(payload, "mount response")
    raw_state = _expect_str(raw, "state")
    try:
        state = MountState(raw_state)
    except ValueError as error:
        raise HotLoadProtocolError(
            f"mount response has unknown state {raw_state!r}"
        ) from error

    raw_error = raw.get("error")
    failure = None
    if raw_error is not None:
        error_body = _expect_mapping(raw_error, "mount error")
        failure = MountFailure(
            code=_expect_str(error_body, "code"),
            message=_expect_str(error_body, "message"),
            retryable=_expect_bool(error_body, "retryable"),
        )

    pinned_source = raw.get("pinned_source")
    if pinned_source is not None and not isinstance(pinned_source, str):
        raise HotLoadProtocolError(
            "mount response field 'pinned_source' must be a string or null"
        )
    return Mount(
        id=_expect_str(raw, "id"),
        source=_expect_str(raw, "source"),
        target=_expect_str(raw, "target"),
        path=_expect_str(raw, "path"),
        pinned_source=pinned_source,
        state=state,
        error=failure,
    )


def _expect_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HotLoadProtocolError(f"Hot Load {name} must be a JSON object")
    return value


def _expect_str(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str):
        raise HotLoadProtocolError(
            f"Hot Load response field {field!r} must be a string"
        )
    return item


def _expect_bool(value: Mapping[str, Any], field: str) -> bool:
    item = value.get(field)
    if not isinstance(item, bool):
        raise HotLoadProtocolError(
            f"Hot Load response field {field!r} must be a boolean"
        )
    return item
