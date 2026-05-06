import enum
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
)

import yaml
from requests import ReadTimeout
from watchfiles import watch

from truss.base.constants import (
    CONFIG_FILE,
    DEFAULT_REMOTE_NAME,
    PRODUCTION_ENVIRONMENT_NAME,
)
from truss.base.truss_config import ModelServer
from truss.local.local_config_handler import LocalConfigHandler
from truss.remote.baseten import custom_types
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi, resolve_rest_api_url
from truss.remote.baseten.auth import ApiKeyCredential, AuthService, OAuthSession
from truss.remote.baseten.core import (
    ChainDeploymentHandleAtomic,
    ModelId,
    ModelIdentifier,
    ModelName,
    ModelVersionHandle,
    ModelVersionId,
    archive_dir,
    create_bis_llm_service,
    create_chain_atomic,
    create_truss_service,
    exists_model,
    get_dev_version_from_versions,
    get_prod_version_from_versions,
    get_truss_watch_state,
    upload_chain_artifact,
    upload_truss,
    validate_truss_config_against_backend,
)
from truss.remote.baseten.error import ApiError, AuthorizationError, RemoteError
from truss.remote.baseten.oauth import OAuthCredential
from truss.remote.baseten.service import BasetenService, URLConfig
from truss.remote.baseten.utils.transfer import base64_encoded_json_str
from truss.remote.truss_remote import RemoteUser, TrussRemote
from truss.templates.control.control.helpers.custom_types import PatchType
from truss.truss_handle import build as truss_build
from truss.truss_handle.truss_handle import TrussHandle
from truss.util.path import is_ignored, load_trussignore_patterns_from_truss_dir

if TYPE_CHECKING:
    from rich import console as rich_console
    from rich import progress


# Server-side cap on raw config.yaml; payloads larger than this are dropped.
RAW_CONFIG_MAX_BYTES = 100 * 1024


class PatchStatus(enum.Enum):
    SUCCESS = enum.auto()
    FAILED = enum.auto()
    SKIPPED = enum.auto()


class PatchResult(NamedTuple):
    status: PatchStatus
    message: str


def retry_patch(
    patch_fn: Callable[[], Optional[PatchResult]],
    console: "rich_console.Console",
    error_console: "rich_console.Console",
    max_retries: int = 5,
    retry_delay_seconds: int = 5,
) -> None:
    for attempt in range(max_retries):
        result = patch_fn()
        if result is not None and result.status in (
            PatchStatus.SUCCESS,
            PatchStatus.SKIPPED,
        ):
            return
        if attempt < max_retries - 1:
            time.sleep(retry_delay_seconds)
        else:
            msg = result.message if result else "Unknown error"
            error_console.print(
                f"Initial sync failed after {max_retries} attempts: {msg}"
            )
            sys.exit(1)


class FinalPushData(custom_types.OracleData):
    class Config:
        protected_namespaces = ()

    is_draft: bool
    model_id: Optional[str]
    preserve_previous_prod_deployment: bool
    origin: Optional[custom_types.ModelOrigin] = None
    environment: Optional[str] = None
    allow_truss_download: bool
    team_id: Optional[str] = None
    labels: Optional[Dict[str, Any]] = None
    raw_config: Optional[bytes] = None


class BasetenRemote(TrussRemote):
    def __init__(
        self,
        remote_url: str,
        api_key: Optional[str] = None,
        *,
        oauth_remote_name: Optional[str] = None,
        oauth_access_token: Optional[str] = None,
        oauth_refresh_token: Optional[str] = None,
        oauth_expires_at: Optional[str] = None,
    ):
        super().__init__(remote_url)
        self._oauth_remote_name = oauth_remote_name
        if oauth_access_token:
            if api_key:
                raise ValueError(
                    f"Remote {oauth_remote_name!r}: cannot specify both api_key "
                    "and OAuth credentials."
                )
            if not (oauth_refresh_token and oauth_expires_at):
                raise ValueError(
                    f"Remote {oauth_remote_name!r}: OAuth credentials require "
                    "access_token, refresh_token, and expires_at."
                )
            self._auth_service = AuthService(
                OAuthSession(
                    api_url=resolve_rest_api_url(remote_url),
                    credential=OAuthCredential(
                        access_token=oauth_access_token,
                        refresh_token=oauth_refresh_token,
                        expires_at=int(oauth_expires_at),
                    ),
                    on_token_refresh=self._persist_refreshed_credential,
                )
            )
        else:
            api_key = api_key or os.environ.get("BASETEN_API_KEY")
            if not api_key:
                raise AuthorizationError("No credentials provided.")
            self._auth_service = AuthService(ApiKeyCredential(api_key=api_key))
        self._api = BasetenApi(remote_url, self._auth_service)

    def fetch_auth_header(self) -> dict[str, str]:
        """Return a fresh ``Authorization`` header for the active credential.

        Call this per-request rather than caching the return value: for OAuth
        credentials the access token is refreshed in-place when near expiry,
        so a stored header can become stale.
        """
        return self._auth_service.fetch_auth_header()

    def _persist_refreshed_credential(self, credential: OAuthCredential) -> None:
        from truss.remote.remote_factory import AuthType, RemoteFactory
        from truss.remote.truss_remote import RemoteConfig

        if not self._oauth_remote_name:
            return
        RemoteFactory.update_remote_config(
            RemoteConfig(
                name=self._oauth_remote_name,
                configs={
                    "remote_provider": DEFAULT_REMOTE_NAME,
                    "remote_url": self.remote_url,
                    "auth_type": AuthType.OAUTH,
                    "oauth_access_token": credential.access_token,
                    "oauth_refresh_token": credential.refresh_token,
                    "oauth_expires_at": str(credential.expires_at),
                },
            )
        )

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

    def get_oidc_info(self) -> custom_types.OidcInfo:
        """Get OIDC configuration information for workload identity."""
        org_id = self._api.get_organization_id()
        teams = self._api.get_teams()
        team_info = [
            custom_types.OidcTeamInfo(id=team.id, name=team.name)
            for team in teams.values()
        ]

        return custom_types.OidcInfo(
            org_id=org_id,
            teams=team_info,
            issuer="https://oidc.baseten.co",
            audience="oidc.baseten.co",
            workload_types=["model_container", "model_build"],
        )

    def _validate_bis_llm_push_options(
        self,
        publish: bool,
        promote: bool,
        preserve_previous_prod_deployment: bool,
        disable_truss_download: bool,
        deployment_name: Optional[str],
        origin: Optional[custom_types.ModelOrigin],
        environment: Optional[str],
        deploy_timeout_minutes: Optional[int],
    ) -> None:
        if not publish:
            raise ValueError(
                "Development deployment is not supported for BIS LLM models."
            )
        if promote:
            raise ValueError("Promotion is not supported for BIS LLM models ")
        if environment:
            raise ValueError("Environment is not supported for BIS LLM models.")
        if preserve_previous_prod_deployment:
            raise ValueError(
                "Preserve previous production deployment is not supported for BIS LLM models."
            )
        if disable_truss_download:
            raise ValueError(
                "Disable truss download is not supported for BIS LLM models."
            )
        if deployment_name:
            raise ValueError("Deployment name is not supported for BIS LLM models.")
        if origin:
            raise ValueError("Origin is not supported for BIS LLM models.")
        if deploy_timeout_minutes is not None:
            raise ValueError(
                "Deploy timeout minutes is not supported for BIS LLM models."
            )

    def _prepare_bis_llm_request_body(
        self,
        config: Any,
        model_name: str,
        model_id: Optional[str],
        labels: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "resources": config.resources.model_dump(exclude_none=True)
        }
        if model_id is None:
            body["name"] = model_name
        if config.environment_variables:
            body["environment_variables"] = config.environment_variables
        if config.weights:
            body["weights"] = config.weights.model_dump(exclude_none=True)
        if labels is not None:
            body["metadata"] = labels

        if config.bis_llm:
            if config.bis_llm.version:
                body["llm_version"] = config.bis_llm.version

            if config.bis_llm.config is not None:
                body["llm_config"] = config.bis_llm.config

            if config.bis_llm.additional_autoscaling_config is not None:
                body["additional_autoscaling_config"] = (
                    config.bis_llm.additional_autoscaling_config.model_dump(
                        exclude_none=True
                    )
                )

        return body

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
        deploy_timeout_minutes: Optional[int] = None,
        team_id: Optional[str] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> FinalPushData:
        if model_name.isspace():
            raise ValueError("Model name cannot be empty")

        if truss_handle.is_scattered():
            truss_handle = TrussHandle(truss_handle.gather())

        config = truss_handle.spec.config

        if config.bis_llm is not None:
            self._validate_bis_llm_push_options(
                publish=publish,
                promote=promote,
                preserve_previous_prod_deployment=preserve_previous_prod_deployment,
                disable_truss_download=disable_truss_download,
                deployment_name=deployment_name,
                origin=origin,
                environment=environment,
                deploy_timeout_minutes=deploy_timeout_minutes,
            )

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

        if deploy_timeout_minutes is not None and (
            deploy_timeout_minutes < 10 or deploy_timeout_minutes > 1440
        ):
            raise ValueError(
                "deploy-timeout-minutes must be between 10 minutes and 1440 minutes (24 hours)"
            )

        model_id = exists_model(self._api, model_name, team_id=team_id)

        if model_id is not None and disable_truss_download:
            raise ValueError("disable-truss-download can only be used for new models")

        config.validate_forbid_extra()
        encoded_config_str = base64_encoded_json_str(config.to_dict())
        if config.bis_llm is None:
            validate_truss_config_against_backend(self._api, encoded_config_str)
        default_config = (truss_handle.truss_dir / CONFIG_FILE).resolve()
        config_yaml_override: Optional[bytes] = None
        if truss_handle.spec.config_path.resolve() != default_config:
            # Match non-override packing: stream literal file bytes into config.yaml.
            config_yaml_override = truss_handle.spec.config_path.read_bytes()
        # Capture the original config.yaml bytes so the server can persist them as-is.
        # Best-effort: if the file cannot be read or is too large, push proceeds without raw.
        raw_config_bytes = config_yaml_override
        if raw_config_bytes is None:
            try:
                raw_config_bytes = truss_handle.spec.config_path.read_bytes()
            except OSError as exc:
                logging.warning(
                    f"Could not read raw config.yaml ({exc}); proceeding without uploading raw config."
                )
        if (
            raw_config_bytes is not None
            and len(raw_config_bytes) > RAW_CONFIG_MAX_BYTES
        ):
            logging.warning(
                f"Raw config.yaml is {len(raw_config_bytes)} bytes, exceeds "
                f"{RAW_CONFIG_MAX_BYTES} byte cap; proceeding without uploading raw config."
            )
            raw_config_bytes = None
        temp_file = archive_dir(
            truss_handle._truss_dir,
            progress_bar,
            config_yaml_override=config_yaml_override,
        )
        s3_key = upload_truss(self._api, temp_file, progress_bar)

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
            team_id=team_id,
            labels=labels,
            raw_config=raw_config_bytes,
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
        deploy_timeout_minutes: Optional[int] = None,
        team_id: Optional[str] = None,
        labels: Optional[Dict[str, Any]] = None,
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
            deploy_timeout_minutes=deploy_timeout_minutes,
            team_id=team_id,
            labels=labels,
        )

        config = truss_handle.spec.config
        if config.bis_llm is not None:
            model_version_handle = create_bis_llm_service(
                api=self._api,
                body=self._prepare_bis_llm_request_body(
                    config=config,
                    model_name=model_name,
                    model_id=push_data.model_id,
                    labels=push_data.labels,
                ),
                model_id=push_data.model_id,
                team_id=push_data.team_id,
            )
            return BasetenService(
                model_version_handle=model_version_handle,
                is_draft=False,
                header_provider=self.fetch_auth_header,
                service_url=f"{self._remote_url}/model_versions/{model_version_handle.version_id}",
                truss_handle=truss_handle,
                api=self._api,
                url_config=URLConfig.BIS_LLM,
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
            deploy_timeout_minutes=deploy_timeout_minutes,
            team_id=push_data.team_id,
            labels=push_data.labels,
            raw_config=push_data.raw_config,
        )

        if model_version_handle.instance_type_name:
            logging.info(
                f"Deploying truss using {model_version_handle.instance_type_name} instance type."
            )

        return BasetenService(
            model_version_handle=model_version_handle,
            is_draft=push_data.is_draft,
            header_provider=self.fetch_auth_header,
            service_url=f"{self._remote_url}/model_versions/{model_version_handle.version_id}",
            truss_handle=truss_handle,
            api=self._api,
            url_config=URLConfig.MODEL,
        )

    def push_chain_atomic(
        self,
        chain_name: str,
        entrypoint_artifact: custom_types.ChainletArtifact,
        dependency_artifacts: List[custom_types.ChainletArtifact],
        truss_user_env: b10_types.TrussUserEnv,
        chain_root: Optional[Path] = None,
        publish: bool = False,
        environment: Optional[str] = None,
        progress_bar: Optional[Type["progress.Progress"]] = None,
        disable_chain_download: bool = False,
        deployment_name: Optional[str] = None,
        team_id: Optional[str] = None,
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
                disable_truss_download=disable_chain_download,
                deployment_name=deployment_name,
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

        # Upload raw chain artifact if chain_root is provided
        raw_chain_s3_key = None
        if chain_root is not None:
            logging.info("Uploading source artifact")
            # Create a tar file from the chain root directory
            original_source_tar = archive_dir(dir=chain_root, progress_bar=progress_bar)
            # Upload the chain artifact to S3
            raw_chain_s3_key = upload_chain_artifact(
                api=self._api,
                serialize_file=original_source_tar,
                progress_bar=progress_bar,
            )
        chain_deployment_handle = create_chain_atomic(
            api=self._api,
            chain_name=chain_name,
            entrypoint=chainlet_data[0],
            dependencies=chainlet_data[1:],
            is_draft=not publish,
            truss_user_env=truss_user_env,
            environment=environment,
            original_source_artifact_s3_key=raw_chain_s3_key,
            allow_truss_download=not disable_chain_download,
            deployment_name=deployment_name,
            team_id=team_id,
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
                    "No development model found. Run `truss push --watch` then try again."
                )
            return dev_version

        # Return the production deployment version.
        prod_version = get_prod_version_from_versions(model_versions)
        if not prod_version:
            raise RemoteError(
                "No production model found. Run `truss push` then try again."
            )
        return prod_version

    def _get_service_url_path_and_model_ids(
        self, api: BasetenApi, model_identifier: ModelIdentifier, published: bool
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
            # Use resolve_model_for_watch to handle team disambiguation
            # Import here to avoid circular import
            from truss.cli.resolvers.model_team_resolver import resolve_model_for_watch

            model, model_versions = resolve_model_for_watch(
                self, model_identifier.value
            )
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
            header_provider=self.fetch_auth_header,
            service_url=f"{self._remote_url}{service_url_path}",
            api=self._api,
        )

    def sync_truss_to_dev_version_by_name(
        self,
        model_name: str,
        target_directory: str,
        console: "rich_console.Console",
        error_console: "rich_console.Console",
        team_name: Optional[str] = None,
    ) -> None:
        # Resolve model with team disambiguation and verify development deployment exists
        # Import here to avoid circular import
        from truss.cli.resolvers.model_team_resolver import resolve_model_for_watch

        model, versions = resolve_model_for_watch(
            self, model_name, provided_team_name=team_name
        )
        self.sync_truss_to_dev_version_with_model(
            model, versions, target_directory, console, error_console
        )

    def sync_truss_to_dev_version_with_model(
        self,
        resolved_model: dict,
        resolved_versions: List[dict],
        target_directory: str,
        console: "rich_console.Console",
        error_console: "rich_console.Console",
        hot_reload: bool = False,
    ) -> None:
        """Sync truss to dev version using pre-resolved model (no team re-prompting)."""
        dev_version = get_dev_version_from_versions(resolved_versions)
        if not dev_version:
            raise RemoteError(
                "No development model found. Run `truss push --watch` then try again."
            )

        watch_path = Path(target_directory)
        truss_ignore_patterns = load_trussignore_patterns_from_truss_dir(watch_path)

        def watch_filter(_, path):
            return not is_ignored(Path(path), truss_ignore_patterns)

        # disable watchfiles logger
        logging.getLogger("watchfiles.main").disabled = True

        console.print(f"🚰 Attempting to sync truss at '{watch_path}' with remote")
        retry_patch(
            patch_fn=lambda: self._patch_with_model(
                watch_path,
                truss_ignore_patterns,
                resolved_model,
                resolved_versions,
                console,
                error_console,
                hot_reload=hot_reload,
            ),
            console=console,
            error_console=error_console,
        )

        # Prepare watch paths including external package directories
        truss_handle = TrussHandle(watch_path)
        watch_paths = [watch_path]

        # Add external package directories to watch list for seamless live development
        if not truss_handle.no_external_packages:
            external_dirs = truss_handle.spec.external_package_dirs_paths
            watch_paths.extend(external_dirs)
            console.print(
                f"👀 Watching for changes to truss at '{watch_path}' "
                f"and {len(external_dirs)} external package director{'y' if len(external_dirs) == 1 else 'ies'}..."
            )
        else:
            console.print(f"👀 Watching for changes to truss at '{watch_path}'...")

        for _ in watch(*watch_paths, watch_filter=watch_filter, raise_interrupt=False):
            logging.debug("Changes detected, creating patch...")
            self._patch_with_model(
                watch_path,
                truss_ignore_patterns,
                resolved_model,
                resolved_versions,
                console,
                error_console,
                hot_reload=hot_reload,
            )

    def _patch(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
        console: Optional["rich_console.Console"] = None,
        resolved_model: Optional[dict] = None,
        resolved_versions: Optional[List[dict]] = None,
        chainlets_only: bool = False,
        hot_reload: bool = False,
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
        if not model_name:
            return PatchResult(
                PatchStatus.FAILED, "Truss config is missing a model name."
            )

        # Use pre-resolved model if provided to avoid re-prompting for team selection
        if resolved_model is not None and resolved_versions is not None:
            model = resolved_model
            versions = resolved_versions
        else:
            try:
                # Use resolve_model_for_watch to handle team disambiguation
                # Import here to avoid circular import
                from truss.cli.resolvers.model_team_resolver import (
                    resolve_model_for_watch,
                )

                model, versions = resolve_model_for_watch(
                    self, model_name, chainlets_only=chainlets_only
                )
            except Exception as e:
                return PatchResult(
                    PatchStatus.FAILED, f"Model not found: {model_name}. {e}"
                )

        model_id = model["id"]
        dev_version = get_dev_version_from_versions(versions)
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

        truss_watch_state = get_truss_watch_state(self._api, model_id)
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

        # Only hot-reload when every patch is a model code change. Mixed patches
        # (e.g. pip requirements + code) fall back to a normal cold restart.
        # The flag is set on each ModelCodePatch body so it flows through the
        # backend opaquely (patch_ops is stored/forwarded as raw JSON).
        is_hot_reload = hot_reload and all(
            p.type == PatchType.MODEL_CODE for p in patch_request.patch_ops
        )
        if is_hot_reload:
            for p in patch_request.patch_ops:
                p.body.hot_reload = True

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
                resp = self._api.patch_draft_truss_two_step(model_id, patch_request)
            else:
                resp = self._api.sync_draft_truss(model_id)
            return resp

        hot_reload_suffix = " (hot reload)" if is_hot_reload else ""

        try:
            if console:
                with console.status(f"Applying patch{hot_reload_suffix}..."):
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
                    "success_message",
                    f"Model {model_name} patched successfully{hot_reload_suffix}.",
                ),
            )

    def patch(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
        console: "rich_console.Console",
        error_console: "rich_console.Console",
    ) -> PatchResult:
        result = self._patch(watch_path, truss_ignore_patterns, console=console)
        if result.status in (PatchStatus.SUCCESS, PatchStatus.SKIPPED):
            console.print(result.message, style="green")
        else:
            error_console.print(result.message)
        return result

    def patch_for_chainlet(
        self, watch_path: Path, truss_ignore_patterns: List[str]
    ) -> PatchResult:
        # Use chainlets_only=True to query chainlet oracles (origin=CHAINS)
        # instead of regular models (origin=BASETEN)
        return self._patch(
            watch_path, truss_ignore_patterns, console=None, chainlets_only=True
        )

    def _patch_with_model(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
        resolved_model: dict,
        resolved_versions: List[dict],
        console: "rich_console.Console",
        error_console: "rich_console.Console",
        hot_reload: bool = False,
    ) -> PatchResult:
        """Patch with pre-resolved model (no team re-prompting)."""
        result = self._patch(
            watch_path,
            truss_ignore_patterns,
            console=console,
            resolved_model=resolved_model,
            resolved_versions=resolved_versions,
            hot_reload=hot_reload,
        )
        if result.status == PatchStatus.SUCCESS:
            console.print(result.message, style="green")
        elif result.status == PatchStatus.SKIPPED:
            logging.debug(result.message)
        else:
            error_console.print(result.message)
        return result

    def upsert_training_project(self, training_project, team_id=None):
        return self._api.upsert_training_project(training_project, team_id=team_id)

    def get_trainer_session(self, session_id):
        return self._api.get_trainer_session(session_id)

    def create_trainer_session(self, training_project_id=None):
        return self._api.create_trainer_session(training_project_id=training_project_id)

    def create_trainer_server(
        self, session_id, model, lora_rank=16, max_seq_len=4096, seed=None
    ):
        return self._api.create_trainer_server(
            session_id=session_id,
            model=model,
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
            seed=seed,
        )

    def deactivate_loop_deployment(self, model_name: str) -> None:
        self._api.deactivate_loop_deployment(model_name)
