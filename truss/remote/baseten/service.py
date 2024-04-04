import time
from typing import Dict, Optional

import requests
from tenacity import retry, stop_after_delay, wait_fixed
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.truss_remote import TrussService
from truss.truss_handle import TrussHandle
from truss.util.errors import RemoteNetworkError

DEFAULT_STREAM_ENCODING = "utf-8"


class BasetenService(TrussService):
    def __init__(
        self,
        model_id: str,
        model_version_id: str,
        is_draft: bool,
        api_key: str,
        service_url: str,
        api: BasetenApi,
        truss_handle: Optional[TrussHandle] = None,
    ):
        super().__init__(is_draft=is_draft, service_url=service_url)
        self._model_id = model_id
        self._model_version_id = model_version_id
        self._auth_service = AuthService(api_key=api_key)
        self._api = api
        self._truss_handle = truss_handle

    def is_live(self) -> bool:
        raise NotImplementedError

    def is_ready(self) -> bool:
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version_id(self) -> str:
        return self._model_version_id

    @property
    def invocation_url(self) -> str:
        return f"{self._service_url}/predict"

    def predict(
        self,
        model_request_body: Dict,
    ):
        response = self._send_request(
            self.invocation_url, "POST", data=model_request_body, stream=True
        )

        if response.headers.get("transfer-encoding") == "chunked":
            # Case of streaming response, the backend does not set an encoding, so
            # manually decode to the contents to utf-8 here.
            def decode_content():
                for chunk in response.iter_content(
                    chunk_size=8192, decode_unicode=True
                ):
                    # Depending on the content-type of the response,
                    # iter_content will either emit a byte stream, or a stream
                    # of strings. Only decode in the bytes case.
                    if isinstance(chunk, bytes):
                        yield chunk.decode(response.encoding or DEFAULT_STREAM_ENCODING)
                    else:
                        yield chunk

            return decode_content()

        parsed_response = response.json()

        if "error" in parsed_response:
            # In the case that the model is in a non-ready state, the response
            # will be a json with an `error` key.
            return parsed_response

        return response.json()["model_output"]

    def authenticate(self) -> dict:
        return self._auth_service.authenticate().header()

    def logs_url(self, base_url: str) -> str:
        return f"{base_url}/models/{self._model_id}/logs/{self._model_version_id}"

    @retry(stop=stop_after_delay(60), wait=wait_fixed(1))
    def _fetch_deployment(self):
        return self._api.get_deployment(self._model_id, self._model_version_id)

    def poll_deployment_status(self):
        """
        Wait for the service to be deployed.
        """
        while True:
            time.sleep(1)
            try:
                deployment = self._fetch_deployment()
                yield deployment["status"]
            except requests.exceptions.RequestException:
                raise RemoteNetworkError("Could not reach backend.")
