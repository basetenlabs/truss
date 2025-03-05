import enum
import time
import urllib.parse
import warnings
from typing import Any, Dict, Iterator, NamedTuple, Optional

import requests
from tenacity import retry, stop_after_delay, wait_fixed

from truss.base.errors import RemoteNetworkError
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.core import ModelVersionHandle
from truss.remote.truss_remote import TrussService
from truss.truss_handle.truss_handle import TrussHandle

# "classes created inside an enum will not become a member" -> intended here anyway.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*enum.*")

DEFAULT_STREAM_ENCODING = "utf-8"


def _add_model_subdomain(rest_api_url: str, model_subdomain: str) -> str:
    """E.g. `https://api.baseten.co` -> `https://{model_subdomain}.api.baseten.co`"""
    parsed_url = urllib.parse.urlparse(rest_api_url)
    new_netloc = f"{model_subdomain}.{parsed_url.netloc}"
    model_url = parsed_url._replace(netloc=new_netloc)
    return str(urllib.parse.urlunparse(model_url))


class URLConfig(enum.Enum):
    class Data(NamedTuple):
        prefix: str
        invoke_endpoint: str
        app_endpoint: str

    MODEL = Data("model", "predict", "models")
    CHAIN = Data("chain", "run_remote", "chains")

    @staticmethod
    def invoke_url(
        hostname: str,  # E.g. https://model-{model_id}.api.baseten.co
        config: "URLConfig",
        entity_version_id: str,
        is_draft,
    ) -> str:
        """Get the URL for the predict/run_remote endpoint."""

        if is_draft:
            return f"{hostname}/development/{config.value.invoke_endpoint}"
        else:
            return f"{hostname}/deployment/{entity_version_id}/{config.value.invoke_endpoint}"

    @staticmethod
    def status_page_url(
        app_url: str,  # E.g. https://app.baseten.co/
        config: "URLConfig",
        entity_id: str,
    ) -> str:
        return f"{app_url}/{config.value.app_endpoint}/{entity_id}/overview"

    @staticmethod
    def model_logs_url(
        app_url: str,  # E.g. https://app.baseten.co/
        model_id: str,
        model_version_id: str,
    ) -> str:
        return (
            f"{app_url}/{URLConfig.MODEL.value.app_endpoint}/{model_id}/logs/"
            f"{model_version_id}"
        )

    @staticmethod
    def chainlet_logs_url(
        app_url: str,  # E.g. https://app.baseten.co/
        chain_id: str,
        chain_deployment_id: str,
        chainlet_id: str,
    ) -> str:
        return (
            f"{app_url}/{URLConfig.CHAIN.value.app_endpoint}/{chain_id}/logs/"
            f"{chain_deployment_id}/{chainlet_id}"
        )


class BasetenService(TrussService):
    def __init__(
        self,
        model_version_handle: ModelVersionHandle,
        is_draft: bool,
        api_key: str,
        service_url: str,
        api: BasetenApi,
        truss_handle: Optional[TrussHandle] = None,
    ):
        super().__init__(is_draft=is_draft, service_url=service_url)
        self._model_version_handle = model_version_handle
        self._auth_service = AuthService(api_key=api_key)
        self._api = api
        self._truss_handle = truss_handle

    def is_live(self) -> bool:
        raise NotImplementedError

    def is_ready(self) -> bool:
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return self._model_version_handle.model_id

    @property
    def model_version_id(self) -> str:
        return self._model_version_handle.version_id

    def predict(self, model_request_body: Dict) -> Any:
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
        return URLConfig.model_logs_url(
            self._api.app_url, self.model_id, self.model_version_id
        )

    @property
    def predict_url(self) -> str:
        handle = self._model_version_handle

        return URLConfig.invoke_url(
            hostname=handle.hostname,
            config=URLConfig.MODEL,
            entity_version_id=handle.version_id,
            is_draft=self.is_draft,
        )

    @retry(stop=stop_after_delay(60), wait=wait_fixed(1), reraise=True)
    def _fetch_deployment(self) -> Any:
        return self._api.get_deployment(self.model_id, self.model_version_id)

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
