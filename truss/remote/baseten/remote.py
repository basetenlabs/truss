from typing import Optional

from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.core import (
    archive_truss,
    create_model,
    exists_model,
    upload_model,
)
from truss.remote.baseten.service import BasetenService
from truss.remote.baseten.utils.transfer import base64_encoded_json_str
from truss.remote.truss_remote import TrussRemote
from truss.truss_handle import TrussHandle


class BasetenRemote(TrussRemote):
    def __init__(self, remote_url: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(remote_url, **kwargs)
        if api_key:
            self._auth_service = AuthService(api_key=api_key)
        else:
            self._auth_service = AuthService()

    def authenticate(self):
        return self._auth_service.validate()

    def push(self, truss_handle: TrussHandle, model_name: str):  # type: ignore
        self.authenticate()
        api = BasetenApi(f"{self._remote_url}/graphql/", self._auth_service)

        if model_name.isspace():
            raise ValueError("Model name cannot be empty")

        if exists_model(api, model_name):
            raise ValueError(f"Model with name {model_name} already exists")

        gathered_truss = TrussHandle(truss_handle.gather())
        encoded_config_str = base64_encoded_json_str(
            gathered_truss._spec._config.to_dict()
        )

        temp_file = archive_truss(gathered_truss)
        s3_key = upload_model(api, temp_file)
        model_id, model_version_id = create_model(
            api=api,
            model_name=model_name,
            s3_key=s3_key,
            config=encoded_config_str,
        )

        return BasetenService(
            model_id=model_id,
            model_version_id=model_version_id,
            is_draft=False,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}/model_versions/{model_version_id}",
            truss_handle=truss_handle,
        )
