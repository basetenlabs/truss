from typing import Dict

from truss.remote.baseten.auth import AuthService
from truss.remote.truss_remote import TrussService
from truss.truss_handle import TrussHandle


class BasetenService(TrussService):
    # TODO(Abu): How to handle liveness/readiness probes from client?
    def __init__(
        self,
        model_id: str,
        model_version_id: str,
        is_draft: bool,
        truss_handle: TrussHandle,
        auth_service: AuthService,
        service_url: str,
    ):

        super().__init__(is_draft=is_draft, service_url=service_url)
        self._model_id = model_id
        self._model_version_id = model_version_id
        self._auth_service = auth_service

    @property
    def is_live(self) -> bool:
        return False

    @property
    def is_ready(self) -> bool:
        return False

    def predict(self, model_request_body: Dict):
        invocation_url = f"{self._service_url}/predict"
        response = self._send_request(invocation_url, "POST", data=model_request_body)
        # TODO(Abu): Do we want to strip response to just be model output? Or keep metadata?
        return response

    def authenticate(self) -> dict:
        return self._auth_service.authenticate().headers()
