import datetime
import json
import logging
import pathlib
import textwrap
from typing import IO, TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Type

from truss.base.errors import ValidationError

if TYPE_CHECKING:
    from rich import progress

from truss.base.constants import PRODUCTION_ENVIRONMENT_NAME
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.util.path import load_trussignore_patterns_from_truss_dir

logger = logging.getLogger(__name__)


DEPLOYING_STATUSES = ["BUILDING", "DEPLOYING", "LOADING_MODEL", "UPDATING"]
ACTIVE_STATUS = "ACTIVE"
NO_ENVIRONMENTS_EXIST_ERROR_MESSAGING = (
    "Model hasn't been deployed yet. No environments exist."
)


class ModelIdentifier:
    value: str


class ModelName(ModelIdentifier):
    def __init__(self, name: str):
        self.value = name


class ModelId(ModelIdentifier):
    def __init__(self, model_id: str):
        self.value = model_id


class ModelVersionId(ModelIdentifier):
    def __init__(self, model_version_id: str):
        self.value = model_version_id


class PatchState(NamedTuple):
    current_hash: str
    current_signature: str


class TrussPatches(NamedTuple):
    django_patch_state: PatchState
    container_patch_state: PatchState


class TrussWatchState(NamedTuple):
    is_container_built_from_push: bool
    patches: Optional[TrussPatches]


class ChainDeploymentHandleAtomic(NamedTuple):
    hostname: str
    chain_id: str
    chain_deployment_id: str
    is_draft: bool


class ModelVersionHandle(NamedTuple):
    version_id: str
    model_id: str
    hostname: str
    instance_type_name: Optional[str] = None


def get_chain_id_by_name(api: BasetenApi, chain_name: str) -> Optional[str]:
    """
    Check if a chain with the given name exists in the Baseten remote.

    Args:
        api: BasetenApi instance
        chain_name: Name of the chain to check for existence

    Returns:
        chain_id if present, otherwise None
    """
    chains = api.get_chains()

    chain_name_id_mapping = {chain["name"]: chain["id"] for chain in chains}
    return chain_name_id_mapping.get(chain_name)


def get_dev_chain_deployment(api: BasetenApi, chain_id: str):
    chain_deployments = api.get_chain_deployments(chain_id)
    dev_deployments = [
        deployment for deployment in chain_deployments if deployment["is_draft"]
    ]
    if not dev_deployments:
        return None
    newest_draft_deployment = max(
        dev_deployments, key=lambda d: datetime.datetime.fromisoformat(d["created"])
    )
    return newest_draft_deployment


def create_chain_atomic(
    api: BasetenApi,
    chain_name: str,
    entrypoint: b10_types.ChainletDataAtomic,
    dependencies: List[b10_types.ChainletDataAtomic],
    is_draft: bool,
    truss_user_env: b10_types.TrussUserEnv,
    environment: Optional[str],
) -> ChainDeploymentHandleAtomic:
    if environment and is_draft:
        logging.info(
            f"Automatically publishing Chain `{chain_name}` based on "
            "environment setting."
        )
        is_draft = False

    chain_id = get_chain_id_by_name(api, chain_name)

    # TODO(Tyron): Refactor for better readability:
    # 1. Prepare all arguments for `deploy_chain_atomic`.
    # 2. Validate argument combinations.
    # 3. Make a single invocation to `deploy_chain_atomic`.

    if is_draft:
        res = api.deploy_chain_atomic(
            entrypoint=entrypoint,
            dependencies=dependencies,
            chain_name=chain_name,
            is_draft=True,
            truss_user_env=truss_user_env,
        )
    elif chain_id:
        # This is the only case where promote has relevance, since
        # if there is no chain already, the first deployment will
        # already be production, and only published deployments can
        # be promoted.
        try:
            res = api.deploy_chain_atomic(
                entrypoint=entrypoint,
                dependencies=dependencies,
                chain_id=chain_id,
                environment=environment,
                truss_user_env=truss_user_env,
            )
        except ApiError as e:
            if (
                e.graphql_error_code
                == BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value
            ):
                raise ValueError(
                    f"Environment `{environment}` does not exist. You can "
                    f"create environments in the Chains UI."
                ) from e

            raise e
    elif environment and environment != PRODUCTION_ENVIRONMENT_NAME:
        raise ValueError(NO_ENVIRONMENTS_EXIST_ERROR_MESSAGING)
    else:
        res = api.deploy_chain_atomic(
            entrypoint=entrypoint,
            dependencies=dependencies,
            chain_name=chain_name,
            truss_user_env=truss_user_env,
        )

    return ChainDeploymentHandleAtomic(
        chain_deployment_id=res["chain_deployment"]["id"],
        chain_id=res["chain_deployment"]["chain"]["id"],
        hostname=res["chain_deployment"]["chain"]["hostname"],
        is_draft=is_draft,
    )


def exists_model(api: BasetenApi, model_name: str) -> Optional[str]:
    """
    Check if a model with the given name exists in the Baseten remote.

    Args:
        api: BasetenApi instance
        model_name: Name of the model to check for existence

    Returns:
        model_id if present, otherwise None
    """
    try:
        model = api.get_model(model_name)
    except ApiError as e:
        if (
            e.graphql_error_code
            == BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value
        ):
            return None

        raise e

    return model["model"]["id"]


def get_model_and_versions(api: BasetenApi, model_name: ModelName) -> Tuple[dict, List]:
    query_result = api.get_model(model_name.value)["model"]
    return query_result, query_result["versions"]


def get_dev_version_from_versions(versions: List[dict]) -> Optional[dict]:
    """Given a list of model version dicts, returns the development version.

    Args:
        versions: List of model version dicts from the Baseten API

    Returns:
        The version in versions corresponding to the development model, or None
        if no development model exists
    """
    for version in versions:
        if version["is_draft"] is True:
            return version
    return None


def get_dev_version(api: BasetenApi, model_name: str) -> Optional[dict]:
    """Queries the Baseten API and returns a dict representing the
    development version for the given model_name.

    Args:
        api: BasetenApi instance
        model_name: Name of the model to get the development version for

    Returns:
        The development version of the model
    """
    model = api.get_model(model_name)
    versions = model["model"]["versions"]
    return get_dev_version_from_versions(versions)


def get_truss_watch_state(api: BasetenApi, model_name: str) -> TrussWatchState:
    response = api.get_truss_watch_state(model_name)["truss_watch_state"]
    django_patch_state = (
        None
        if response["django_patch_state"] is None
        else PatchState(
            current_hash=response["django_patch_state"]["current_hash"],
            current_signature=response["django_patch_state"]["current_signature"],
        )
    )
    container_patch_state = (
        None
        if response["container_patch_state"] is None
        else PatchState(
            current_hash=response["container_patch_state"]["current_hash"],
            current_signature=response["container_patch_state"]["current_signature"],
        )
    )
    patches = None
    if django_patch_state and container_patch_state:
        patches = TrussPatches(
            django_patch_state=django_patch_state,
            container_patch_state=container_patch_state,
        )
    return TrussWatchState(
        is_container_built_from_push=response["is_container_built_from_push"],
        patches=patches,
    )


def get_prod_version_from_versions(versions: List[dict]) -> Optional[dict]:
    # Loop over versions instead of using the primary_version field because
    # primary_version is set to the development version ID if no published
    # models exist.
    for version in versions:
        if version["is_primary"]:
            return version
    return None


def archive_dir(
    dir: pathlib.Path, progress_bar: Optional[Type["progress.Progress"]] = None
) -> IO:
    """Archive a TrussHandle into a tar file.

    Returns:
        A file-like object containing the tar file
    """
    # check for a truss_ignore file and read the ignore patterns if it exists
    ignore_patterns = load_trussignore_patterns_from_truss_dir(dir)

    try:
        temp_file = create_tar_with_progress_bar(
            dir, ignore_patterns, progress_bar=progress_bar
        )
    except PermissionError:
        # workaround for Windows bug with Tempfile that causes PermissionErrors
        temp_file = create_tar_with_progress_bar(
            dir, ignore_patterns, delete=False, progress_bar=progress_bar
        )
    temp_file.file.seek(0)
    return temp_file


def upload_truss(
    api: BasetenApi,
    serialize_file: IO,
    progress_bar: Optional[Type["progress.Progress"]],
) -> str:
    """
    Upload a TrussHandle to the Baseten remote.

    Args:
        api: BasetenApi instance
        serialize_file: File-like object containing the serialized TrussHandle

    Returns:
        The S3 key of the uploaded file
    """
    temp_credentials_s3_upload = api.model_s3_upload_credentials()
    s3_key = temp_credentials_s3_upload.pop("s3_key")
    s3_bucket = temp_credentials_s3_upload.pop("s3_bucket")
    multipart_upload_boto3(
        serialize_file.name, s3_bucket, s3_key, temp_credentials_s3_upload, progress_bar
    )
    return s3_key


def create_truss_service(
    api: BasetenApi,
    model_name: str,
    s3_key: str,
    config: str,
    truss_user_env: b10_types.TrussUserEnv,
    semver_bump: str = "MINOR",
    preserve_previous_prod_deployment: bool = False,
    allow_truss_download: bool = False,
    is_draft: Optional[bool] = False,
    model_id: Optional[str] = None,
    deployment_name: Optional[str] = None,
    origin: Optional[b10_types.ModelOrigin] = None,
    environment: Optional[str] = None,
    preserve_env_instance_type: bool = False,
) -> ModelVersionHandle:
    """
    Create a model in the Baseten remote.

    Args:
        api: BasetenApi instance.
        model_name: Name of the model to create.
        s3_key: S3 key of the uploaded TrussHandle.
        config: Base64 encoded JSON string of the Truss config.
        semver_bump: Semver bump type, defaults to "MINOR".
        promote: Whether to promote the model after deploy, defaults to False.
        preserve_previous_prod_deployment: Whether to scale old production deployment
            to zero.
        deployment_name: Name to apply to the created deployment. Not applied to
            development model.

    Returns:
        A Model Version handle.
    """
    if is_draft:
        model_version_json = api.create_development_model_from_truss(
            model_name,
            s3_key,
            config,
            truss_user_env,
            allow_truss_download=allow_truss_download,
            origin=origin,
        )

        return ModelVersionHandle(
            version_id=model_version_json["id"],
            model_id=model_version_json["oracle"]["id"],
            hostname=model_version_json["oracle"]["hostname"],
            instance_type_name=(
                model_version_json["instance_type"]["name"]
                if "instance_type" in model_version_json
                else None
            ),
        )

    if model_id is None:
        if environment and environment != PRODUCTION_ENVIRONMENT_NAME:
            raise ValueError(NO_ENVIRONMENTS_EXIST_ERROR_MESSAGING)

        model_version_json = api.create_model_from_truss(
            model_name,
            s3_key,
            config,
            semver_bump,
            truss_user_env,
            allow_truss_download=allow_truss_download,
            deployment_name=deployment_name,
            origin=origin,
        )

        return ModelVersionHandle(
            version_id=model_version_json["id"],
            model_id=model_version_json["oracle"]["id"],
            hostname=model_version_json["oracle"]["hostname"],
            instance_type_name=(
                model_version_json["instance_type"]["name"]
                if "instance_type" in model_version_json
                else None
            ),
        )

    try:
        model_version_json = api.create_model_version_from_truss(
            model_id,
            s3_key,
            config,
            semver_bump,
            truss_user_env,
            preserve_previous_prod_deployment=preserve_previous_prod_deployment,
            deployment_name=deployment_name,
            environment=environment,
            preserve_env_instance_type=preserve_env_instance_type,
        )
    except ApiError as e:
        if (
            e.graphql_error_code
            == BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value
        ):
            raise ValueError(
                f"Environment `{environment}` does not exist. You can create "
                "environments in the Baseten UI."
            ) from e
        raise e

    return ModelVersionHandle(
        version_id=model_version_json["id"],
        model_id=model_id,
        hostname=model_version_json["oracle"]["hostname"],
        instance_type_name=(
            model_version_json["instance_type"]["name"]
            if "instance_type" in model_version_json
            else None
        ),
    )


def validate_truss_config(api: BasetenApi, config: str):
    """
    Validate a truss config as well as the truss version.

    Args:
        api: BasetenApi instance
        config: Base64 encoded JSON string of the Truss config

    Returns:
        None if the config is valid, otherwise raises an error message
    """
    valid_config = api.validate_truss(config)
    if not valid_config.get("success"):
        details = json.loads(valid_config.get("details"))
        errors = details.get("errors", [])
        if errors:
            error_messages = "\n".join(textwrap.indent(error, "  ") for error in errors)
            raise ValidationError(
                f"Validation failed with the following errors:\n{error_messages}"
            )
