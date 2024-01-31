import asyncio
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import click
import rich
import yaml
from requests import ReadTimeout
from truss.cli.console import console
from truss.local.local_config_handler import LocalConfigHandler
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.core import (
    ModelId,
    ModelIdentifier,
    ModelName,
    ModelVersionId,
    archive_truss,
    create_truss_service,
    exists_model,
    get_dev_version,
    get_dev_version_from_versions,
    get_model_versions,
    get_prod_version_from_versions,
    upload_truss,
)
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.service import BasetenService
from truss.remote.baseten.utils.transfer import base64_encoded_json_str
from truss.remote.truss_remote import TrussRemote, TrussService
from truss.truss_config import ModelServer
from truss.truss_handle import TrussHandle
from truss.util.path import is_ignored, load_trussignore_patterns
from watchfiles import awatch

logger = logging.getLogger(__name__)


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
        promote: bool = False,
        preserve_previous_prod_deployment: bool = False,
        deployment_name: Optional[str] = None,
    ):
        if model_name.isspace():
            raise ValueError("Model name cannot be empty")

        model_id = exists_model(self._api, model_name)

        gathered_truss = TrussHandle(truss_handle.gather())
        if gathered_truss.spec.model_server != ModelServer.TrussServer:
            publish = True

        if promote:
            # If we are promoting a model after deploy, it must be published.
            # Draft models cannot be promoted.
            publish = True

        if not publish and deployment_name:
            raise ValueError(
                "Deployment name cannot be used for development deployment"
            )

        if not promote and preserve_previous_prod_deployment:
            raise ValueError(
                "preserve-previous-production-deployment can only be used with the '--promote' option"
            )

        if deployment_name and not re.match(r"^[0-9a-zA-Z_\-\.]*$", deployment_name):
            raise ValueError(
                "Deployment name must only contain alphanumeric, -, _ and . characters"
            )

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
            promote=promote,
            preserve_previous_prod_deployment=preserve_previous_prod_deployment,
            deployment_name=deployment_name,
        )

        return BasetenService(
            model_id=model_id,
            model_version_id=model_version_id,
            is_draft=not publish,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}/model_versions/{model_version_id}",
            truss_handle=truss_handle,
        )

    @staticmethod
    def _get_matching_version(model_versions: List[dict], published: bool) -> dict:
        if not published:
            # Return the development model version.
            dev_version = get_dev_version_from_versions(model_versions)
            if not dev_version:
                raise click.UsageError(
                    "No development model found. Run `truss push` then try again."
                )
            return dev_version

        # Return the production deployment version.
        prod_version = get_prod_version_from_versions(model_versions)
        if not prod_version:
            raise click.UsageError(
                "No production model found. Run `truss push --publish` then try again."
            )
        return prod_version

    @staticmethod
    def _get_service_url_path_and_model_ids(
        api: BasetenApi, model_identifier: ModelIdentifier, published: bool
    ) -> Tuple[str, str, str]:
        if isinstance(model_identifier, ModelVersionId):
            try:
                model_version = api.get_model_version_by_id(model_identifier.value)
            except ApiError:
                raise click.UsageError(
                    f"Model version {model_identifier.value} not found."
                )
            model_version_id = model_version["model_version"]["id"]
            model_id = model_version["model_version"]["oracle"]["id"]
            service_url_path = f"/model_versions/{model_version_id}"
            return service_url_path, model_id, model_version_id

        if isinstance(model_identifier, ModelName):
            model_id, model_versions = get_model_versions(api, model_identifier)
            model_version = BasetenRemote._get_matching_version(
                model_versions, published
            )
            model_version_id = model_version["id"]
            service_url_path = f"/model_versions/{model_version_id}"
        elif isinstance(model_identifier, ModelId):
            # TODO(helen): consider making this consistent with getting the
            # service via model_name / respect --published in service_url_path.
            try:
                model = api.get_model_by_id(model_identifier.value)
            except ApiError:
                raise click.UsageError(f"Model {model_identifier.value} not found.")
            model_id = model["model"]["id"]
            model_version_id = model["model"]["primary_version"]["id"]
            service_url_path = f"/models/{model_id}"
        else:
            # Model identifier is of invalid type.
            raise click.UsageError(
                "You must either be inside of a Truss directory, or provide --model-deployment or --model options."
            )

        return service_url_path, model_id, model_version_id

    def get_service(self, **kwargs) -> BasetenService:
        try:
            model_identifier = kwargs["model_identifier"]
        except KeyError:
            raise ValueError("Baseten Service requires a model_identifier")

        published = kwargs.get("published", False)
        (
            service_url_path,
            model_id,
            model_version_id,
        ) = self._get_service_url_path_and_model_ids(
            self._api, model_identifier, published
        )

        return BasetenService(
            model_id=model_id,
            model_version_id=model_version_id,
            is_draft=not published,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}{service_url_path}",
        )

    def get_remote_logs_url(
        self,
        service: TrussService,
    ) -> str:
        return service.logs_url(self._remote_url)

    async def _watch(self, watch_path: Path, truss_ignore_patterns: List[str]):
        """Watches for changes in watch_path and sends patch requests for detected changes.

        Changes are detected asynchronously using watchfiles.awatch."""

        def watch_filter(_, path):
            return not is_ignored(
                Path(path),
                truss_ignore_patterns,
            )

        async for _ in awatch(
            watch_path, watch_filter=watch_filter, raise_interrupt=False
        ):
            self.patch(watch_path)

    def sync_truss_to_dev_version_by_name(
        self,
        model_name: str,
        target_directory: str,
    ):
        # verify that development deployment exists for given model name
        dev_version = get_dev_version(
            self._api, model_name
        )  # pylint: disable=protected-access
        if not dev_version:
            raise click.UsageError(
                "No development model found. Run `truss push` then try again."
            )

        watch_path = Path(target_directory)

        rich.print(f"🚰 Attempting to sync truss at '{watch_path}' with remote")
        self.patch(watch_path)

        # disable watchfiles logger
        logging.getLogger("watchfiles.main").disabled = True

        rich.print(f"👀 Watching for changes to truss at '{watch_path}' ...")
        truss_ignore_patterns = load_trussignore_patterns()
        asyncio.run(self._watch(watch_path, truss_ignore_patterns))

    def patch(
        self,
        watch_path: Path,
    ):
        try:
            truss_handle = TrussHandle(watch_path)
        except yaml.parser.ParserError:
            console.print("Unable to parse config file", style="red")
            return
        except ValueError:
            console.print(
                f"Error when reading truss from directory {watch_path}", style="red"
            )
            return
        model_name = truss_handle.spec.config.model_name
        dev_version = get_dev_version(self._api, model_name)  # type: ignore
        if not dev_version:
            console.print(
                f"No development deployment found with model name: {model_name}",
                style="red",
            )
            return
        truss_hash = dev_version.get("truss_hash", None)
        truss_signature = dev_version.get("truss_signature", None)
        if not (truss_hash and truss_signature):
            console.print(
                """Failed to inspect a running remote deployment to watch for changes.
Ensure that there exists a running remote deployment before attempting to watch for changes
            """,
                style="red",
            )
            return
        LocalConfigHandler.add_signature(truss_hash, truss_signature)
        try:
            patch_request = truss_handle.calc_patch(truss_hash)
        except Exception:
            console.print("Failed to calculate patch, bailing on patching", style="red")
            return
        if patch_request:
            if (
                patch_request.prev_hash == patch_request.next_hash
                or len(patch_request.patch_ops) == 0
            ):
                console.print("No changes observed, skipping patching")
                return
            try:
                console.print("Applying patch...")
                resp = self._api.patch_draft_truss(model_name, patch_request)
            except ReadTimeout:
                console.print(
                    "Read Timeout when attempting to connect to remote. Bailing on patching",
                    style="red",
                )
                return
            except Exception:
                console.print(
                    "Failed to patch draft deployment, bailing on patching", style="red"
                )
                return
            if not resp["succeeded"]:
                needs_full_deploy = resp.get("needs_full_deploy", None)
                if needs_full_deploy:
                    console.print(
                        f"Model {model_name} is not able to be patched, use `truss push` to deploy",
                        style="red",
                    )
                else:
                    console.print(
                        f"Failed to patch: `{resp['error']}`. Model left in original state",
                        style="red",
                    )
            else:
                console.print(
                    resp.get(
                        "success_message",
                        f"Model {model_name} patched successfully",
                    ),
                    style="green",
                )
