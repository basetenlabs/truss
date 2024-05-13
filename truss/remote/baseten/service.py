import time
import urllib.parse
from typing import Any, Dict, Iterator, Optional

import requests
from tenacity import retry, stop_after_delay, wait_fixed
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.truss_remote import TrussService
from truss.truss_handle import TrussHandle
from truss.util.errors import RemoteNetworkError

DEFAULT_STREAM_ENCODING = "utf-8"


def _add_model_subdomain(rest_api_url: str, model_subdomain: str) -> str:
    """E.g. `https://api.baseten.co` -> `https://{model_subdomain}.api.baseten.co`"""
    parsed_url = urllib.parse.urlparse(rest_api_url)
    new_netloc = f"{model_subdomain}.{parsed_url.netloc}"
    model_url = parsed_url._replace(netloc=new_netloc)
    return str(urllib.parse.urlunparse(model_url))


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
    ) -> Any:
        response = self._send_request(
            self.predict_url, "POST", data=model_request_body, stream=True
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

        return response.json()

    def authenticate(self) -> dict:
        return self._auth_service.authenticate().header()

    @property
    def logs_url(self) -> str:
        return (
            f"{self._api.remote_url}/models/{self._model_id}/"
            f"logs/{self._model_version_id}"
        )

    @property
    def predict_url(self) -> str:
        """
        Get the URL for the prediction endpoint.
        """
        # E.g. `https://api.baseten.co` -> `https://model-{model_id}.api.baseten.co`
        url = _add_model_subdomain(self._api.rest_api_url, f"model-{self.model_id}")
        if self.is_draft:
            # "https://model-{model_id}.api.baseten.co/development".
            url = f"{url}/development/predict"
        else:
            # "https://model-{model_id}.api.baseten.co/deployment/{deployment_id}".
            url = f"{url}/deployment/{self.model_version_id}/predict"
        return url

    @retry(stop=stop_after_delay(60), wait=wait_fixed(1), reraise=True)
    def _fetch_deployment(self) -> Any:
        return self._api.get_deployment(self._model_id, self._model_version_id)

    def poll_deployment_status(self, sleep_secs: int = 1) -> Iterator[str]:
        """
        Wait for the service to be deployed.
        """
        while True:
            time.sleep(sleep_secs)
            try:
                deployment = self._fetch_deployment()
                yield deployment["status"]
            except requests.exceptions.RequestException:
                raise RemoteNetworkError("Could not reach backend.")
