import abc
import functools
from typing import Type, TypeVar

import httpx
from slay import definitions


def _handle_respose(response: httpx.Response):
    # TODO: improve error handling, extract context from response and include in
    # re-raised exception. Consider re-raising same exception or if not a use a
    # generic "RPCError" exception class or similar.
    if response.is_server_error:
        raise ValueError(response)
    if response.is_client_error:
        raise ValueError(response)
    return response.json()


class BasetenSession:
    """Helper to invoke predict method on baseten deployments."""

    # TODO: make timeout, retries etc. configurable.
    def __init__(self, url: str, api_key: str) -> None:
        self._auth_header = {"Authorization": f"Api-Key {api_key}"}
        self._url = url

    @functools.cached_property
    def _client_sync(self) -> httpx.Client:
        return httpx.Client(base_url=self._url, headers=self._auth_header)

    @functools.cached_property
    def _client_async(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._url, headers=self._auth_header)

    def predict_sync(self, json_paylod):
        return _handle_respose(
            self._client_sync.post(definitions.PREDICT_ENDPOINT, json=json_paylod)
        )

    async def predict_async(self, json_paylod):
        return _handle_respose(
            await self._client_async.post(
                definitions.PREDICT_ENDPOINT, json=json_paylod
            )
        )


class StubBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, url: str, api_key: str) -> None:
        ...


StubT = TypeVar("StubT", bound=StubBase)


def stub_factory(stub_cls: Type[StubT], context: definitions.Context) -> StubT:
    return stub_cls(
        url=context.get_stub_url(stub_cls.__name__),
        api_key=context.get_baseten_api_key(),
    )
