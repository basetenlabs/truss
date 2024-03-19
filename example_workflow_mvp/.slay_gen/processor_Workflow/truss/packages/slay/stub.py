import abc
import functools
from typing import Type, TypeVar

import httpx
from slay import definitions

_PREDICT_ENDPOINT = "/predict"


class BasetenSession:
    def __init__(self, url: str, api_key: str):
        self._auth_header = {"Authorization": f"Api-Key {api_key}"}
        self._url = url

    @functools.cached_property
    def _client_sync(self):
        return httpx.Client(base_url=self._url, headers=self._auth_header)

    @functools.cached_property
    def _client_async(self):
        return httpx.AsyncClient(base_url=self._url, headers=self._auth_header)

    def predict_sync(self, json_paylod):
        response = self._client_sync.post(_PREDICT_ENDPOINT, json=json_paylod)
        return response.json()

    async def predict_async(self, json_paylod):
        response = await self._client_async.post(_PREDICT_ENDPOINT, json=json_paylod)
        return response.json()


class StubBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, url: str, api_key: str) -> None:
        ...


StubT = TypeVar("StubT", bound=StubBase)


def stub_factory(stub_cls: Type[StubT], context: definitions.Context) -> StubT:
    return stub_cls(
        context.get_stub_url(stub_cls.__name__), context.get_baseten_api_key()
    )
