"""Client for the pod-local BDN Hot Load API.

Hot Load gives an opted-in model pod a read-only front door at ``/bdn``: the
Unix socket ``/bdn/hotload.sock`` and the directory ``/bdn/mounts``. A request
names a BDN volume source and a target directory name; the node attaches the
resolved, immutable volume view at ``/bdn/mounts/<target>`` before it answers,
so a successful attach is readable as soon as it returns.

The wire contract is HTTP+JSON over the socket:

* ``POST /v1/hotload/volumes`` with ``{"source", "target", "include"?,
  "exclude"?}`` and a required ``Idempotency-Key`` header returns ``202`` and
  the volume attachment. Replaying the same key with the same body returns the
  same attachment; the same key with a different body is ``409``.
* ``GET /v1/hotload/volumes`` returns ``{"volumes": [...]}``.
* ``GET``/``DELETE /v1/hotload/volumes/{id}`` read or detach one attachment.
* Errors carry ``{"code", "message", "retryable"}`` in the body.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

DEFAULT_SOCKET_PATH = Path("/bdn/hotload.sock")
MOUNT_ROOT = Path("/bdn/mounts")
_VOLUMES_PATH = "/v1/hotload/volumes"


class AttachmentState(str, Enum):
    READY = "READY"
    FAILED = "FAILED"


@dataclass(frozen=True)
class VolumeAttachment:
    """One attached volume view, as the node reports it."""

    id: str
    revision: int
    source: str
    target: str
    path: str
    pinned_source: str
    state: AttachmentState
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


class HotLoadError(Exception):
    """Base class for Hot Load client errors."""


class HotLoadConnectionError(HotLoadError):
    """The local Hot Load socket could not complete a request."""


class HotLoadProtocolError(HotLoadError):
    """The server returned a response that does not match the API contract."""


class HotLoadAPIError(HotLoadError):
    """The server rejected a request; ``code`` is the stable error identifier."""

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


class HotLoadAttachError(HotLoadError):
    """The server accepted the request but reports the attachment as failed."""

    def __init__(self, attachment: VolumeAttachment) -> None:
        super().__init__(
            f"Hot Load attachment {attachment.id} for {attachment.source} at "
            f"{attachment.path} is {attachment.state.value}"
        )
        self.attachment = attachment


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
        request_timeout_sec: float = 600.0,
        retry_interval_sec: float = 0.1,
        max_retries: int = 2,
    ) -> None:
        # An attach resolves the source, acquires the shared view, and binds it
        # before answering, so the request timeout bounds a whole cold attach,
        # not a round trip.
        if request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive")
        if retry_interval_sec < 0:
            raise ValueError("retry_interval_sec must not be negative")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.socket_path = Path(socket_path)
        self.request_timeout_sec = request_timeout_sec
        self.retry_interval_sec = retry_interval_sec
        self.max_retries = max_retries

    def attach(
        self,
        source: str,
        target: str,
        *,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        idempotency_key: Optional[str] = None,
    ) -> VolumeAttachment:
        """Attach ``source`` at ``/bdn/mounts/<target>`` and return it once readable.

        ``idempotency_key`` defaults to a fresh key that is reused across the
        client's own transport retries, so a retried request cannot attach the
        same volume twice. Pass a stable key to make the call safe to repeat
        from the caller's side too.
        """
        key = idempotency_key or uuid.uuid4().hex
        body: dict[str, Any] = {"source": source, "target": target}
        if include:
            body["include"] = list(include)
        if exclude:
            body["exclude"] = list(exclude)
        attachment = _parse_attachment(
            self._request(
                "POST",
                _VOLUMES_PATH,
                body=body,
                headers={"Idempotency-Key": key},
                retry_transport=True,
            )
        )
        if attachment.state is not AttachmentState.READY:
            raise HotLoadAttachError(attachment)
        return attachment

    def list_volumes(self) -> list[VolumeAttachment]:
        payload = _expect_mapping(self._request("GET", _VOLUMES_PATH), "volume list")
        volumes = payload.get("volumes")
        if not isinstance(volumes, list):
            raise HotLoadProtocolError(
                "volume list response must contain a volumes array"
            )
        return [_parse_attachment(volume) for volume in volumes]

    def get_volume(self, attachment_id: str) -> VolumeAttachment:
        return _parse_attachment(self._request("GET", _volume_path(attachment_id)))

    def detach(self, attachment_id: str) -> None:
        """Detach one attachment. The server answers once the mount is gone."""
        self._request("DELETE", _volume_path(attachment_id))

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
                time.sleep(self.retry_interval_sec)
                continue
            finally:
                connection.close()

            payload = _decode_json(response_body) if response_body else None
            if 200 <= response.status < 300:
                return payload
            api_error = _parse_api_error(response.status, payload)
            if not (retry_request and api_error.retryable and attempt + 1 < attempts):
                raise api_error
            time.sleep(self.retry_interval_sec)
        raise RuntimeError("unreachable Hot Load request retry state")


def _volume_path(attachment_id: str) -> str:
    if not attachment_id or "/" in attachment_id:
        raise ValueError("attachment_id must be a non-empty path segment")
    return f"{_VOLUMES_PATH}/{attachment_id}"


def _decode_json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HotLoadProtocolError("Hot Load returned invalid JSON") from error


def _parse_api_error(status_code: int, payload: Any) -> HotLoadAPIError:
    error = _expect_mapping(payload, "error response")
    return HotLoadAPIError(
        status_code=status_code,
        code=_expect_str(error, "code"),
        message=_expect_str(error, "message"),
        retryable=_expect_bool(error, "retryable"),
    )


def _parse_attachment(payload: Any) -> VolumeAttachment:
    raw = _expect_mapping(payload, "volume attachment")
    raw_state = _expect_str(raw, "state")
    try:
        state = AttachmentState(raw_state)
    except ValueError as error:
        raise HotLoadProtocolError(
            f"volume attachment has unknown state {raw_state!r}"
        ) from error
    return VolumeAttachment(
        id=_expect_str(raw, "id"),
        revision=_expect_int(raw, "revision"),
        source=_expect_str(raw, "source"),
        target=_expect_str(raw, "target"),
        path=_expect_str(raw, "path"),
        pinned_source=_expect_str(raw, "pinned_source"),
        state=state,
        include=_expect_string_tuple(raw, "include"),
        exclude=_expect_string_tuple(raw, "exclude"),
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


def _expect_int(value: Mapping[str, Any], field: str) -> int:
    item = value.get(field)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise HotLoadProtocolError(
            f"Hot Load response field {field!r} must be a non-negative integer"
        )
    return item


def _expect_string_tuple(value: Mapping[str, Any], field: str) -> tuple[str, ...]:
    item = value.get(field, [])
    if not isinstance(item, list) or not all(isinstance(entry, str) for entry in item):
        raise HotLoadProtocolError(
            f"Hot Load response field {field!r} must be an array of strings"
        )
    return tuple(item)
