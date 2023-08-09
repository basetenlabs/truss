import logging
from pathlib import Path

import yaml
from truss.contexts.local_loader.truss_file_syncer import TrussFilesSyncer
from truss.local.local_config_handler import LocalConfigHandler
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.core import (
    archive_truss,
    create_truss_service,
    exists_model,
    get_dev_version_info,
    get_model_versions_info,
    upload_truss,
)
from truss.remote.baseten.service import BasetenService
from truss.remote.baseten.utils.transfer import base64_encoded_json_str
from truss.remote.truss_remote import TrussRemote
from truss.truss_config import ModelServer
from truss.truss_handle import TrussHandle


class BasetenRemote(TrussRemote):
    def __init__(self, remote_url: str, api_key: str, **kwargs):
        super().__init__(remote_url, **kwargs)
        self._auth_service = AuthService(api_key=api_key)
        self._api = BasetenApi(f"{self._remote_url}/graphql/", self._auth_service)

    def authenticate(self):
        return self._auth_service.validate()

    def push(  # type: ignore
        self,
        truss_handle: TrussHandle,
        model_name: str,
        publish: bool = True,
        trusted: bool = False,
    ):
        if model_name.isspace():
            raise ValueError("Model name cannot be empty")

        model_id = exists_model(self._api, model_name)

        gathered_truss = TrussHandle(truss_handle.gather())
        if gathered_truss.spec.model_server != ModelServer.TrussServer:
            publish = True
        encoded_config_str = base64_encoded_json_str(
            gathered_truss._spec._config.to_dict()
        )

        temp_file = archive_truss(gathered_truss)
        s3_key = upload_truss(self._api, temp_file)

        model_id, model_version_id = create_truss_service(
            api=self._api,
            model_name=model_name,
            s3_key=s3_key,
            config=encoded_config_str,
            is_draft=not publish,
            model_id=model_id,
            is_trusted=trusted,
        )

        return BasetenService(
            model_id=model_id,
            model_version_id=model_version_id,
            is_draft=not publish,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}/model_versions/{model_version_id}",
            truss_handle=truss_handle,
        )

    def get_baseten_service(
        self, model_name: str, published: bool = False
    ) -> BasetenService:
        model_id, model_versions = get_model_versions_info(self._api, model_name)
        model_version = None
        if published:
            for mv in model_versions:
                if not mv["is_draft"]:
                    model_version = mv
                    break
        else:
            for mv in model_versions:
                if mv["is_draft"]:
                    model_version = mv
                    break
        if model_version is None:
            raise ValueError(
                "No appropriate model version found. Run `truss push` then try again."
            )
        model_version_id = model_version["id"]
        return BasetenService(
            model_id=model_id,
            model_version_id=model_version_id,
            is_draft=not published,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}/model_versions/{model_version_id}",
        )

    def get_remote_logs_url(
        self,
        model_name: str,
        published: bool = False,
    ) -> str:
        service = self.get_baseten_service(model_name, published)
        return f"{self._remote_url}/models/{service._model_id}/versions/{service._model_version_id}/logs"

    def sync_truss_to_dev_version_by_name(
        self,
        model_name: str,
        target_directory: str,
    ):
        # verify that development deployment exists for given model name
        _ = get_dev_version_info(
            self._api, model_name  # pylint: disable=protected-access
        )
        TrussFilesSyncer(
            Path(target_directory),
            self,
        ).start()

        # Since the `TrussFilesSyncer` runs a daemon thread, we run this infinite loop on the main
        # thread to keep it alive. When this loop is interrupted by the user, then the whole process
        # can shutdown gracefully.
        while True:
            pass

    def patch(
        self,
        watch_path: Path,
        logger: logging.Logger,
    ):
        try:
            truss_handle = TrussHandle(watch_path)
        except yaml.parser.ParserError:
            logger.error("Unable to parse config file")
            return
        model_name = truss_handle.spec.config.model_name
        dev_version = get_dev_version_info(self._api, model_name)  # type: ignore
        truss_hash = dev_version.get("truss_hash", None)
        truss_signature = dev_version.get("truss_signature", None)
        LocalConfigHandler.add_signature(truss_hash, truss_signature)
        try:
            patch_request = truss_handle.calc_patch(truss_hash)
        except Exception:
            logger.error("Failed to calculate patch")
            return
        if patch_request:
            if (
                patch_request.prev_hash == patch_request.next_hash
                or len(patch_request.patch_ops) == 0
            ):
                logger.info("No changes observed, skipping deployment")
                return
            resp = self._api.patch_draft_truss(model_name, patch_request)
            if not resp["succeeded"]:
                needs_full_deploy = resp.get("needs_full_deploy", None)
                if needs_full_deploy:
                    logger.info(
                        f"Model {model_name} is not able to be patched, use `truss push` to deploy"
                    )
                else:
                    logger.error(
                        f"Failed to patch: `{resp['error']}`. Model left in original state"
                    )
            else:
                logger.info(
                    resp.get(
                        "success_message",
                        f"Model {model_name} patched successfully.",
                    )
                )
