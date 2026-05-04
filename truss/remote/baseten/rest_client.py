from typing import Any, Callable, Dict

import requests


class RestAPIClient:
    base_url: str

    def __init__(self, base_url: str, header_provider: Callable[[], Dict[str, str]]):
        self.base_url = base_url
        self._header_provider = header_provider
        self.suppress_error_print = False

    def _handle_error(self, resp: requests.Response):
        if 400 <= resp.status_code < 500:
            try:
                data = resp.json()
                if "message" in data and not self.suppress_error_print:
                    print(f"Client error: {data['message']}")
            except ValueError:
                pass
        resp.raise_for_status()

    def get(self, path: str, url_params: Dict[str, str] = {}):
        resp = requests.get(
            f"{self.base_url}/{path}",
            headers=self._header_provider(),
            params=url_params,
        )
        self._handle_error(resp)
        return resp.json()

    def post(self, path: str, body: Any):
        resp = requests.post(
            f"{self.base_url}/{path}", headers=self._header_provider(), json=body
        )
        self._handle_error(resp)
        return resp.json()

    def delete(self, path: str):
        resp = requests.delete(
            f"{self.base_url}/{path}", headers=self._header_provider()
        )
        self._handle_error(resp)
        return resp.json()

    def patch(self, path: str, body: Any):
        resp = requests.patch(
            f"{self.base_url}/{path}", headers=self._header_provider(), json=body
        )
        self._handle_error(resp)
        return resp.json()
