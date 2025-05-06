import enum
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Type

import yaml
from requests import ReadTimeout
from watchfiles import watch

from truss.base.constants import PRODUCTION_ENVIRONMENT_NAME
from truss.base.truss_config import ModelServer
from truss.local.local_config_handler import LocalConfigHandler
from truss.remote.baseten import custom_types
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.core import (
    ChainDeploymentHandleAtomic,
    ModelId,
    ModelIdentifier,
    ModelName,
    ModelVersionHandle,
    ModelVersionId,
    archive_dir,
    create_chain_atomic,
    create_truss_service,
    exists_model,
    get_dev_version,
    get_dev_version_from_versions,
    get_model_and_versions,
    get_prod_version_from_versions,
    get_truss_watch_state,
    upload_truss,
    validate_truss_config,
)
from truss.remote.baseten.error import ApiError, RemoteError
from truss.remote.baseten.service import BasetenService, URLConfig
from truss.remote.baseten.utils.transfer import base64_encoded_json_str
from truss.remote.truss_remote import RemoteUser, TrussRemote
from truss.truss_handle import build as truss_build
from truss.truss_handle.truss_handle import TrussHandle
from truss.util.path import is_ignored, load_trussignore_patterns_from_truss_dir

if TYPE_CHECKING:
    from rich import console as rich_console
    from rich import progress


class PatchStatus(enum.Enum):
    SUCCESS = enum.auto()
    FAILED = enum.auto()
    SKIPPED = enum.auto()


class PatchResult(NamedTuple):
    status: PatchStatus
    message: str


class FinalPushData(custom_types.OracleData):
    class Config:
        protected_namespaces = ()

    is_draft: bool
    model_id: Optional[str]
    preserve_previous_prod_deployment: bool
    origin: Optional[custom_types.ModelOrigin] = None
    environment: Optional[str] = None
    allow_truss_download: bool


class BasetenRemote(TrussRemote):
    def __init__(self, remote_url: str, api_key: str):
        super().__init__(remote_url)
        self._auth_service = AuthService(api_key=api_key)
        self._api = BasetenApi(remote_url, self._auth_service)

    @property
    def api(self) -> BasetenApi:
        return self._api

    def get_chainlets(
        self, chain_deployment_id: str
    ) -> List[custom_types.DeployedChainlet]:
        return [
            custom_types.DeployedChainlet(
                name=chainlet["name"],
                is_entrypoint=chainlet["is_entrypoint"],
                is_draft=chainlet["oracle_version"]["is_draft"],
                status=chainlet["oracle_version"]["current_model_deployment_status"][
                    "status"
                ],
                logs_url=URLConfig.chainlet_logs_url(
                    self.remote_url,
                    chainlet["chain"]["id"],
                    chain_deployment_id,
                    chainlet["id"],
                ),
                oracle_name=chainlet["oracle"]["name"],
            )
            for chainlet in self._api.get_chainlets_by_deployment_id(
                chain_deployment_id
            )
        ]

    def whoami(self) -> RemoteUser:
        resp = self._api._post_graphql_query(
            "query{organization{workspace_name}user{email}}"
        )
        workspace_name = resp["data"]["organization"]["workspace_name"]
        user_email = resp["data"]["user"]["email"]
        return RemoteUser(workspace_name, user_email)

    # Validate and finalize options.
    # Upload Truss files to S3 and return S3 key.
    def _prepare_push(
        self,
        truss_handle: TrussHandle,
        model_name: str,
        publish: bool = True,
        promote: bool = False,
        preserve_previous_prod_deployment: bool = False,
        disable_truss_download: bool = False,
        deployment_name: Optional[str] = None,
        origin: Optional[custom_types.ModelOrigin] = None,
        environment: Optional[str] = None,
        progress_bar: Optional[Type["progress.Progress"]] = None,
    ) -> FinalPushData:
        if model_name.isspace():
            raise ValueError("Model name cannot be empty")

        if truss_handle.is_scattered():
            truss_handle = TrussHandle(truss_handle.gather())

        if truss_handle.spec.model_server != ModelServer.TrussServer:
            publish = True

        if promote:
            environment = PRODUCTION_ENVIRONMENT_NAME

        # If there is a target environment, it must be published.
        # Draft models cannot be promoted.
        if environment and not publish:
            logging.info(
                f"Automatically publishing model '{model_name}' based on environment setting."
            )
            publish = True

        if not publish and deployment_name:
            raise ValueError(
                "Deployment name cannot be used for development deployment"
            )

        if not promote and preserve_previous_prod_deployment:
            raise ValueError(
                "preserve-previous-production-deployment can only be used "
                "with the '--promote' option"
            )

        if deployment_name and not re.match(r"^[0-9a-zA-Z_\-\.]*$", deployment_name):
            raise ValueError(
                "Deployment name must only contain alphanumeric, -, _ and . characters"
            )

        model_id = exists_model(self._api, model_name)

        if model_id is not None and disable_truss_download:
            raise ValueError("disable-truss-download can only be used for new models")

        temp_file = archive_dir(truss_handle._truss_dir, progress_bar)
        s3_key = upload_truss(self._api, temp_file, progress_bar)
        encoded_config_str = base64_encoded_json_str(
            truss_handle._spec._config.to_dict()
        )

        validate_truss_config(self._api, encoded_config_str)

        return FinalPushData(
            model_name=model_name,
            s3_key=s3_key,
            encoded_config_str=encoded_config_str,
            is_draft=not publish,
            model_id=model_id,
            preserve_previous_prod_deployment=preserve_previous_prod_deployment,
            version_name=deployment_name,
            origin=origin,
            environment=environment,
            allow_truss_download=not disable_truss_download,
        )

    def push(  # type: ignore
        self,
        truss_handle: TrussHandle,
        model_name: str,
        working_dir: Path,
        publish: bool = True,
        promote: bool = False,
        preserve_previous_prod_deployment: bool = False,
        disable_truss_download: bool = False,
        deployment_name: Optional[str] = None,
        origin: Optional[custom_types.ModelOrigin] = None,
        environment: Optional[str] = None,
        progress_bar: Optional[Type["progress.Progress"]] = None,
        include_git_info: bool = False,
        preserve_env_instance_type: bool = True,
    ) -> BasetenService:
        push_data = self._prepare_push(
            truss_handle=truss_handle,
            model_name=model_name,
            publish=publish,
            promote=promote,
            preserve_previous_prod_deployment=preserve_previous_prod_deployment,
            disable_truss_download=disable_truss_download,
            deployment_name=deployment_name,
            origin=origin,
            environment=environment,
            progress_bar=progress_bar,
        )

        if include_git_info:
            truss_user_env = b10_types.TrussUserEnv.collect_with_git_info(working_dir)
        else:
            truss_user_env = b10_types.TrussUserEnv.collect()

        # TODO(Tyron): This set of args is duplicated across
        # many functions. We should consolidate them into a
        # data class with standardized default values so
        # we're not drilling these arguments everywhere.
        model_version_handle = create_truss_service(
            api=self._api,
            model_name=push_data.model_name,
            s3_key=push_data.s3_key,
            config=push_data.encoded_config_str,
            is_draft=push_data.is_draft,
            model_id=push_data.model_id,
            preserve_previous_prod_deployment=push_data.preserve_previous_prod_deployment,
            allow_truss_download=push_data.allow_truss_download,
            deployment_name=push_data.version_name,
            origin=push_data.origin,
            environment=push_data.environment,
            truss_user_env=truss_user_env,
            preserve_env_instance_type=preserve_env_instance_type,
        )

        if model_version_handle.instance_type_name:
            logging.info(
                f"Deploying truss using {model_version_handle.instance_type_name} instance type."
            )

        return BasetenService(
            model_version_handle=model_version_handle,
            is_draft=push_data.is_draft,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}/model_versions/{model_version_handle.version_id}",
            truss_handle=truss_handle,
            api=self._api,
        )

    def push_chain_atomic(
        self,
        chain_name: str,
        entrypoint_artifact: custom_types.ChainletArtifact,
        dependency_artifacts: List[custom_types.ChainletArtifact],
        truss_user_env: b10_types.TrussUserEnv,
        publish: bool = False,
        environment: Optional[str] = None,
        progress_bar: Optional[Type["progress.Progress"]] = None,
    ) -> ChainDeploymentHandleAtomic:
        # If we are promoting a model to an environment after deploy, it must be published.
        # Draft models cannot be promoted.
        if environment and not publish:
            publish = True

        chainlet_data: List[custom_types.ChainletDataAtomic] = []

        for artifact in [entrypoint_artifact, *dependency_artifacts]:
            truss_handle = truss_build.load(str(artifact.truss_dir))
            model_name = truss_handle.spec.config.model_name
            assert model_name, "Per creation of artifacts should not be empty."

            push_data = self._prepare_push(
                truss_handle=truss_handle,
                model_name=model_name,
                publish=publish,
                origin=custom_types.ModelOrigin.CHAINS,
                progress_bar=progress_bar,
            )
            oracle_data = custom_types.OracleData(
                model_name=push_data.model_name,
                s3_key=push_data.s3_key,
                encoded_config_str=push_data.encoded_config_str,
                is_draft=push_data.is_draft,
                model_id=push_data.model_id,
                version_name=push_data.version_name,
            )
            chainlet_data.append(
                custom_types.ChainletDataAtomic(
                    name=artifact.display_name, oracle=oracle_data
                )
            )

        chain_deployment_handle = create_chain_atomic(
            api=self._api,
            chain_name=chain_name,
            entrypoint=chainlet_data[0],
            dependencies=chainlet_data[1:],
            is_draft=not publish,
            truss_user_env=truss_user_env,
            environment=environment,
        )
        logging.info("Successfully pushed to baseten. Chain is building and deploying.")
        return chain_deployment_handle

    @staticmethod
    def _get_matching_version(model_versions: List[dict], published: bool) -> dict:
        if not published:
            # Return the development model version.
            dev_version = get_dev_version_from_versions(model_versions)
            if not dev_version:
                raise RemoteError(
                    "No development model found. Run `truss push` then try again."
                )
            return dev_version

        # Return the production deployment version.
        prod_version = get_prod_version_from_versions(model_versions)
        if not prod_version:
            raise RemoteError(
                "No production model found. Run `truss push --publish` then try again."
            )
        return prod_version

    @staticmethod
    def _get_service_url_path_and_model_ids(
        api: BasetenApi, model_identifier: ModelIdentifier, published: bool
    ) -> Tuple[str, ModelVersionHandle]:
        if isinstance(model_identifier, ModelVersionId):
            try:
                model_version = api.get_model_version_by_id(model_identifier.value)
            except ApiError:
                raise RemoteError(f"Model version {model_identifier.value} not found.")
            model_version_id = model_version["model_version"]["id"]
            hostname = model_version["model_version"]["oracle"]["hostname"]
            model_id = model_version["model_version"]["oracle"]["id"]
            service_url_path = f"/model_versions/{model_version_id}"

            return service_url_path, ModelVersionHandle(
                version_id=model_version_id, model_id=model_id, hostname=hostname
            )

        if isinstance(model_identifier, ModelName):
            model, model_versions = get_model_and_versions(api, model_identifier)
            model_version = BasetenRemote._get_matching_version(
                model_versions, published
            )
            model_id = model["id"]
            model_version_id = model_version["id"]
            hostname = model["hostname"]
            service_url_path = f"/model_versions/{model_version_id}"
        elif isinstance(model_identifier, ModelId):
            # TODO(helen): consider making this consistent with getting the
            # service via model_name / respect --published in service_url_path.
            try:
                model = api.get_model_by_id(model_identifier.value)
            except ApiError:
                raise RemoteError(f"Model {model_identifier.value} not found.")
            model_id = model["model"]["id"]
            model_version_id = model["model"]["primary_version"]["id"]
            hostname = model["model"]["hostname"]
            service_url_path = f"/models/{model_id}"
        else:
            # Model identifier is of invalid type.
            raise RemoteError(
                "You must either be inside of a Truss directory, or provide "
                "--model-deployment or --model options."
            )

        return service_url_path, ModelVersionHandle(
            version_id=model_version_id, model_id=model_id, hostname=hostname
        )

    def get_service(self, **kwargs) -> BasetenService:
        try:
            model_identifier = kwargs["model_identifier"]
        except KeyError:
            raise ValueError("Baseten Service requires a model_identifier")

        published = kwargs.get("published", False)
        (service_url_path, model_version_handle) = (
            self._get_service_url_path_and_model_ids(
                self._api, model_identifier, published
            )
        )

        return BasetenService(
            model_version_handle=model_version_handle,
            is_draft=not published,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}{service_url_path}",
            api=self._api,
        )

    def sync_truss_to_dev_version_by_name(
        self,
        model_name: str,
        target_directory: str,
        console: "rich_console.Console",
        error_console: "rich_console.Console",
    ) -> None:
        # verify that development deployment exists for given model name
        dev_version = get_dev_version(self._api, model_name)  # pylint: disable=protected-access
        if not dev_version:
            raise RemoteError(
                "No development model found. Run `truss push` then try again."
            )

        watch_path = Path(target_directory)
        truss_ignore_patterns = load_trussignore_patterns_from_truss_dir(watch_path)

        def watch_filter(_, path):
            return not is_ignored(Path(path), truss_ignore_patterns)

        # disable watchfiles logger
        logging.getLogger("watchfiles.main").disabled = True

        console.print(f"🚰 Attempting to sync truss at '{watch_path}' with remote")
        self.patch(watch_path, truss_ignore_patterns, console, error_console)

        console.print(f"👀 Watching for changes to truss at '{watch_path}' ...")
        for _ in watch(watch_path, watch_filter=watch_filter, raise_interrupt=False):
            self.patch(watch_path, truss_ignore_patterns, console, error_console)

    def _patch(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
        console: Optional["rich_console.Console"] = None,
    ) -> PatchResult:
        try:
            truss_handle = TrussHandle(watch_path)
        except yaml.parser.ParserError as e:
            return PatchResult(PatchStatus.FAILED, f"Unable to parse config file. {e}")
        except ValueError as e:
            return PatchResult(
                PatchStatus.FAILED,
                f"Error when reading truss from directory {watch_path}. {e}",
            )

        model_name = truss_handle.spec.config.model_name
        dev_version = get_dev_version(self._api, model_name)  # type: ignore
        if not dev_version:
            return PatchResult(
                PatchStatus.FAILED,
                f"No development deployment found for model: {model_name}.",
            )

        truss_hash = dev_version.get("truss_hash", None)
        truss_signature = dev_version.get("truss_signature", None)
        if not (truss_hash and truss_signature):
            return PatchResult(
                PatchStatus.FAILED,
                (
                    "Failed to inspect a running remote deployment to watch for "
                    "changes.  Ensure that there exists a running remote deployment"
                    " before attempting to watch for changes."
                ),
            )

        truss_watch_state = get_truss_watch_state(self._api, model_name)  # type: ignore
        # Make sure the patches are calculated against the current django patch state, if it exists.
        # This is important to ensure that the sequence of patches for a given sesion forms a
        # valid patch sequence (via a linked list)
        if truss_watch_state.patches:
            truss_hash = truss_watch_state.patches.django_patch_state.current_hash
            truss_signature = (
                truss_watch_state.patches.django_patch_state.current_signature
            )
            logging.debug(f"db patch hash: {truss_hash}")
            logging.debug(
                f"container_patch_hash: {truss_watch_state.patches.container_patch_state.current_hash}"
            )
        LocalConfigHandler.add_signature(truss_hash, truss_signature)
        try:
            patch_request = truss_handle.calc_patch(truss_hash, truss_ignore_patterns)
        except Exception as e:
            return PatchResult(PatchStatus.FAILED, f"Failed to calculate patch. {e}")
        if not patch_request:
            return PatchResult(
                PatchStatus.FAILED,
                "Failed to calculate patch. Change type might not be supported.",
            )

        django_has_unapplied_patches = (
            not truss_watch_state.is_container_built_from_push
            and truss_watch_state.patches
            and (
                truss_watch_state.patches.django_patch_state.current_hash
                != truss_watch_state.patches.container_patch_state.current_hash
            )
        )
        should_create_patch = (
            patch_request.prev_hash != patch_request.next_hash
            and len(patch_request.patch_ops) > 0
        )
        is_synced = not django_has_unapplied_patches and not should_create_patch
        if is_synced:
            return PatchResult(
                PatchStatus.SKIPPED, "No changes observed, skipping patching."
            )

        def do_patch():
            if should_create_patch:
                resp = self._api.patch_draft_truss_two_step(model_name, patch_request)
            else:
                resp = self._api.sync_draft_truss(model_name)
            return resp

        try:
            if console:
                with console.status("Applying patch..."):
                    resp = do_patch()
            else:
                resp = do_patch()

        except ReadTimeout:
            return PatchResult(
                PatchStatus.FAILED, "Read Timeout when attempting to patch remote."
            )
        except Exception as e:
            return PatchResult(
                PatchStatus.FAILED, f"Failed to patch draft deployment. {e}"
            )
        if not resp["succeeded"]:
            needs_full_deploy = resp.get("needs_full_deploy", None)
            if needs_full_deploy:
                message = (
                    f"Model {model_name} is not able to be patched: `{resp['error']}`. "
                    f"Use `truss push` to deploy."
                )
            else:
                message = (
                    f"Failed to patch. Server error: `{resp['error']}`. "
                    "Model left in original state."
                )
            return PatchResult(PatchStatus.FAILED, message)
        else:
            return PatchResult(
                PatchStatus.SUCCESS,
                resp.get(
                    "success_message", f"Model {model_name} patched successfully."
                ),
            )

    def patch(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
        console: "rich_console.Console",
        error_console: "rich_console.Console",
    ):
        result = self._patch(watch_path, truss_ignore_patterns)
        if result.status in (PatchStatus.SUCCESS, PatchStatus.SKIPPED):
            console.print(result.message, style="green")
        else:
            error_console.print(result.message)

    def patch_for_chainlet(
        self, watch_path: Path, truss_ignore_patterns: List[str]
    ) -> PatchResult:
        return self._patch(watch_path, truss_ignore_patterns, console=None)

    def upsert_training_project(self, training_project):
        return self._api.upsert_training_project(training_project)
