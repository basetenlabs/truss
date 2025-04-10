from typing import Any, Dict

import requests


class RestAPIClient:
    base_url: str
    headers: Dict[str, str]

    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url
        self.headers = headers

    def _handle_error(self, resp: requests.Response):
        if 400 <= resp.status_code < 500:
            try:
                data = resp.json()
                if "message" in data:
                    print(f"Client error: {data['message']}")
            except ValueError:
                pass
        resp.raise_for_status()

    def get(self, path: str, url_params: Dict[str, str] = {}):
        resp = requests.get(
            f"{self.base_url}/{path}", headers=self.headers, params=url_params
        )
        self._handle_error(resp)
        return resp.json()

    def post(self, path: str, body: Any):
        resp = requests.post(f"{self.base_url}/{path}", headers=self.headers, json=body)
        self._handle_error(resp)
        return resp.json()

    def delete(self, path: str):
        resp = requests.delete(f"{self.base_url}/{path}", headers=self.headers)
        self._handle_error(resp)
        return resp.json()
