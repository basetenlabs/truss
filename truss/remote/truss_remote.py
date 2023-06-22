from abc import ABC, abstractmethod
from typing import Dict, Optional

import requests
from truss.truss_handle import TrussHandle


class TrussService(ABC):
    def __init__(self, service_url: str, is_draft: bool, **kwargs):
        self._service_url = service_url
        self._is_draft = is_draft

    def _send_request(
        self,
        url: str,
        method: str,
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ):
        if not headers:
            headers = {}

        auth_header = self.authenticate()
        headers = {**headers, **auth_header}
        if method == "GET":
            return requests.request(method, url, headers=headers)
        elif method == "POST":
            if not data:
                raise ValueError("POST request must have data")
            return requests.request(method, url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")

    @property
    def is_draft(self):
        return self._is_draft

    @property
    def is_live(self) -> bool:
        liveness_url = f"{self._service_url}/"
        response = self._send_request(liveness_url, "GET", {})
        return response.status_code == 200

    @property
    def is_ready(self) -> bool:
        readiness_url = f"{self._service_url}/v1/models/model"
        response = self._send_request(readiness_url, "GET", {})
        return response.status_code == 200

    def predict(self, model_request_body: Dict):  # -> Response:
        invocation_url = f"{self._service_url}/v1/models/model:predict"
        response = self._send_request(invocation_url, "POST", model_request_body)
        return response

    def patch(self):
        raise NotImplementedError

    @abstractmethod
    def authenticate(self) -> dict:
        return {}


class TrussRemote(ABC):
    def __init__(self, remote_url: str, **kwargs):
        self._remote_url = remote_url

    @abstractmethod
    def push(self, truss_handle: TrussHandle, **kwargs):
        pass

    @abstractmethod
    def authenticate(self, **kwargs):
        pass
