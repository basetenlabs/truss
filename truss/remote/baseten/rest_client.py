from typing import Any, Callable

import requests

from truss.remote.baseten.user_agent import with_user_agent


class RestAPIClient:
    base_url: str

    def __init__(self, base_url: str, header_provider: Callable[[], dict[str, str]]):
        self.base_url = base_url
        self._header_provider = header_provider
        self.suppress_error_print = False

    def _headers(self) -> dict[str, str]:
        return with_user_agent(self._header_provider())

    def _handle_error(self, resp: requests.Response):
        if 400 <= resp.status_code < 500:
            try:
                data = resp.json()
                if "message" in data and not self.suppress_error_print:
                    print(f"Client error: {data['message']}")
            except ValueError:
                pass
        resp.raise_for_status()

    def get(self, path: str, url_params: dict[str, str] = {}):
        resp = requests.get(
            f"{self.base_url}/{path}", headers=self._headers(), params=url_params
        )
        self._handle_error(resp)
        return resp.json()

    def post(self, path: str, body: Any):
        resp = requests.post(
            f"{self.base_url}/{path}", headers=self._headers(), json=body
        )
        self._handle_error(resp)
        return resp.json()

    def delete(self, path: str):
        resp = requests.delete(f"{self.base_url}/{path}", headers=self._headers())
        self._handle_error(resp)
        return resp.json()

    def patch(self, path: str, body: Any):
        resp = requests.patch(
            f"{self.base_url}/{path}", headers=self._headers(), json=body
        )
        self._handle_error(resp)
        return resp.json()
