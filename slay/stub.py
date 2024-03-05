import functools

import httpx


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
        response = self._client_sync.post("/predict", json=json_paylod)
        return response.json()

    async def predict_async(self, json_paylod):
        response = await self._client_async.post("/predict", json=json_paylod)
        return response.json()


class StubBase:
    ...
