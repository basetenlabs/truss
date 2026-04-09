"""Sync and async HTTP clients for the queue server."""

from __future__ import annotations

import os
from types import TracebackType

import httpx
from pydantic import TypeAdapter

from thinker_client.models import (
    EnqueueRequest,
    EnqueueResponse,
    GetOpStatusesRequest,
    GetOpStatusesResponse,
    Operation,
    OperationStatus,
)

_OperationAdapter = TypeAdapter(Operation)


def _baseten_auth_headers(base_url: str, api_key: str | None = None) -> dict[str, str]:
    key = api_key if api_key is not None else os.getenv("TRM_API_KEY")
    if key:
        return {"Authorization": f"Api-Key {key}"}
    return {}


class QueueClient:
    """Synchronous client for the queue server."""

    def __init__(self, base_url: str, client: httpx.Client | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        auth_headers = _baseten_auth_headers(self._base_url, api_key)
        if client is not None:
            self._client = client
            if auth_headers:
                self._client.headers.update(auth_headers)
            self._owns_client = False
        else:
            self._client = httpx.Client(headers=auth_headers)
            self._owns_client = True

    # -- context manager --------------------------------------------------

    def __enter__(self) -> QueueClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- endpoints --------------------------------------------------------

    def health(self) -> None:
        resp = self._client.get(f"{self._base_url}/health")
        resp.raise_for_status()

    def enqueue_ops(self, ops: list[Operation]) -> list[str]:
        request = EnqueueRequest(ops=ops)
        resp = self._client.post(
            f"{self._base_url}/enqueue_ops",
            json=request.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return EnqueueResponse.model_validate(resp.json()).op_ids

    def peek_op(self) -> Operation | None:
        resp = self._client.post(f"{self._base_url}/peek_op")
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            return None
        return _OperationAdapter.validate_python(data)

    def pop_op(self) -> Operation | None:
        resp = self._client.post(f"{self._base_url}/pop_op")
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            return None
        return _OperationAdapter.validate_python(data)

    def get_op_statuses(self, op_ids: list[str]) -> list[OperationStatus]:
        request = GetOpStatusesRequest(op_ids=op_ids)
        resp = self._client.post(
            f"{self._base_url}/get_op_statuses",
            json=request.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return GetOpStatusesResponse.model_validate(resp.json()).statuses

    def update_op_status(self, status: OperationStatus) -> OperationStatus:
        resp = self._client.post(
            f"{self._base_url}/update_op_status",
            json=status.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return OperationStatus.model_validate(resp.json())


class AsyncQueueClient:
    """Asynchronous client for the queue server."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        auth_headers = _baseten_auth_headers(self._base_url, api_key)
        if client is not None:
            self._client = client
            if auth_headers:
                self._client.headers.update(auth_headers)
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(headers=auth_headers)
            self._owns_client = True

    # -- context manager --------------------------------------------------

    async def __aenter__(self) -> AsyncQueueClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- endpoints --------------------------------------------------------

    async def health(self) -> None:
        resp = await self._client.get(f"{self._base_url}/health")
        resp.raise_for_status()

    async def enqueue_ops(self, ops: list[Operation]) -> list[str]:
        request = EnqueueRequest(ops=ops)
        resp = await self._client.post(
            f"{self._base_url}/enqueue_ops",
            json=request.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return EnqueueResponse.model_validate(resp.json()).op_ids

    async def peek_op(self) -> Operation | None:
        resp = await self._client.post(f"{self._base_url}/peek_op")
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            return None
        return _OperationAdapter.validate_python(data)

    async def pop_op(self) -> Operation | None:
        resp = await self._client.post(f"{self._base_url}/pop_op")
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            return None
        return _OperationAdapter.validate_python(data)

    async def get_op_statuses(self, op_ids: list[str]) -> list[OperationStatus]:
        request = GetOpStatusesRequest(op_ids=op_ids)
        resp = await self._client.post(
            f"{self._base_url}/get_op_statuses",
            json=request.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return GetOpStatusesResponse.model_validate(resp.json()).statuses

    async def update_op_status(self, status: OperationStatus) -> OperationStatus:
        resp = await self._client.post(
            f"{self._base_url}/update_op_status",
            json=status.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return OperationStatus.model_validate(resp.json())
