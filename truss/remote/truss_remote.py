from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Union

import requests

if TYPE_CHECKING:
    from rich import console as rich_console
from truss.truss_handle.truss_handle import TrussHandle


class RemoteUser:
    """Class to hold information about the remote user"""

    workspace_name: str
    user_email: str

    def __init__(self, workspace_name: str, user_email: str):
        self.workspace_name = workspace_name
        self.user_email = user_email


class TrussService(ABC):
    """
    Define the abstract base class for a TrussService.

    A TrussService interacts with a service at a given URL and can either be in
    draft or non-draft mode.

    Attributes:
        _service_url: The URL of the service.
        _is_draft: A boolean indicating if the service is in draft mode.

    Args:
        service_url: The URL of the service.
        is_draft: A boolean indicating if the service is in draft mode.
        **kwargs: Additional keyword arguments to initialize the service.

    """

    def __init__(self, service_url: str, is_draft: bool, **kwargs) -> None:
        self._service_url = service_url
        self._is_draft = is_draft

    def _send_request(
        self,
        url: str,
        method: str,
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None,
        stream: Optional[bool] = False,
    ) -> Any:
        """
        Send a HTTP request.

        Args:
            url: The URL to send the request to.
            method: The HTTP method to use.
            headers: Optional dictionary of headers to include in the request.
            data: Optional dictionary of data to include in the request.

        Raises:
            ValueError: If an unsupported method is used or a POST request has no data.

        Returns:
            A Response object resulting from the request.

        """
        if not headers:
            headers = {}

        auth_header = self.authenticate()
        headers = {**headers, **auth_header}
        if method == "GET":
            response = requests.request(method, url, headers=headers, stream=stream)
        elif method == "POST":
            if data is None:
                raise ValueError("POST request must have data")
            response = requests.request(
                method, url, json=data, headers=headers, stream=stream
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 401:
            raise ValueError(
                f"Authentication failed with status code {response.status_code}"
            )

        return response

    @property
    def is_draft(self) -> bool:
        """
        Check if the service is in draft mode.

        Returns:
            A boolean indicating if the service is in draft mode.
        """
        return self._is_draft

    @abstractmethod
    def is_live(self) -> bool:
        """
        Check if the service is live.

        Sends a GET request to the root of the service and returns whether it
        is successful.

        Returns:
            A boolean indicating if the service is live.
        """
        liveness_url = f"{self._service_url}/"
        response = self._send_request(liveness_url, "GET", {})
        return response.status_code == 200

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the service is ready.

        Sends a GET request to the model path of the service and returns whether it
        is successful.

        Returns:
            A boolean indicating if the service is ready.
        """
        readiness_url = f"{self._service_url}/v1/models/model"
        response = self._send_request(readiness_url, "GET", {})
        return response.status_code == 200

    def predict(self, model_request_body: Dict) -> Any:
        """
        Send a prediction request to the service.

        Args:
            model_request_body: A dictionary representing the body of the
            prediction request.

        Returns:
            A Response object resulting from the prediction request.
        """
        return self._send_request(self.predict_url, "POST", data=model_request_body)

    def patch(self) -> None:
        """
        Patch the service. TrussServices in draft mode can be patched.
        """
        raise NotImplementedError

    @abstractmethod
    def authenticate(self) -> dict:
        """
        Authenticate to the service.

        This method should be implemented in subclasses and return a dictionary
        of headers to include in requests to the service with authentication
        information.
        """
        return {}

    @property
    @abstractmethod
    def logs_url(self) -> str:
        """
        Get the URL for the service logs.
        """
        pass

    @property
    @abstractmethod
    def predict_url(self) -> str:
        """
        Get the URL for the prediction endpoint.
        """
        pass

    @abstractmethod
    def poll_deployment_status(self, sleep_secs: int = 1) -> Iterator[str]:
        """
        Poll for a deployment status.
        """
        pass


class TrussRemote(ABC):
    """
    Define the abstract base class for a remote Truss service.

    A remote Truss service is a service that can push a TrussHandle to a remote
    location. The `push` and `authenticate` methods should be implemented in subclasses.

    Attributes:
        _remote_url: The URL of the remote service.

    Args:
        remote_url: The URL of the remote service.
        **kwargs: Additional keyword arguments needed to initialize the remote service.

    """

    def __init__(self, remote_url: str) -> None:
        self._remote_url = remote_url

    @property
    def remote_url(self) -> str:
        return self._remote_url

    @abstractmethod
    def push(self, truss_handle: TrussHandle, **kwargs) -> TrussService:
        """
        Push a TrussHandle to the remote service.

        This method should be implemented in subclasses and return a TrussService
        object for interacting with the remote service.

        Args:
            truss_handle: The TrussHandle to push to the remote service.
            **kwargs: Additional keyword arguments for the push operation.

        """

    @abstractmethod
    def whoami(self) -> RemoteUser:
        """
        Returns account information for the current user.

        This method should be implemented in subclasses and return a RemoteUser.


        """

    @abstractmethod
    def get_service(self, **kwargs) -> TrussService:
        """
        Get a TrussService object for interacting with the remote service.

        This method should be implemented in subclasses and return a TrussService
        object for interacting with the remote service. This support keyword args,
        so that implementing remotes can be flexible with how a service is instantiated.

        Args:
            **kwargs: Keyword arguments for the get_service operation.

        """

    @abstractmethod
    def sync_truss_to_dev_version_by_name(
        self,
        model_name: str,
        target_directory: str,
        console: "rich_console.Console",
        error_console: "rich_console.Console",
    ) -> None:
        """
        This method watches for changes to files in the `target_directory`,
        and syncs them to the development version of the model, identified
        by name.

        Args:
            model_name: The name of the model to sync to the dev version
            target_directory: The directory to sync the model to
            console: For printing informative output.
            error_console: For printing errors.
        """


@dataclass
class RemoteConfig:
    """Class to hold configs for various remotes"""

    name: str
    configs: Dict = field(default_factory=dict)

    # TODO: use a better, safer lib/schema for configs, this is terrible.
    @staticmethod
    def parse_bool(bool_or_str: Union[str, bool]) -> bool:
        if isinstance(bool_or_str, bool):
            return bool_or_str
        elif bool_or_str.lower() == "true":
            return True
        elif bool_or_str.lower() == "false":
            return False
        else:
            raise ValueError(
                f"Expected a string denoting a boolean, got `{bool_or_str}`."
            )
