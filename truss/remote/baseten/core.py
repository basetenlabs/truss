import logging
import typing
from typing import IO, List, Optional, Tuple

import truss
from truss.remote.baseten import types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.truss_handle import TrussHandle
from truss.util.path import load_trussignore_patterns

logger = logging.getLogger(__name__)


DEPLOYING_STATUSES = ["BUILDING", "DEPLOYING", "LOADING_MODEL", "UPDATING"]
ACTIVE_STATUS = "ACTIVE"


class ChainDeploymentHandle(typing.NamedTuple):
    chain_id: str
    chain_deployment_id: str


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


def create_chain(
    api: BasetenApi,
    chain_id: Optional[str],
    chain_name: str,
    chainlets: List[b10_types.ChainletData],
    is_draft: bool = False,
) -> ChainDeploymentHandle:
    if is_draft:
        response = api.deploy_draft_chain(chain_name, chainlets)
    elif chain_id:
        response = api.deploy_chain_deployment(chain_id, chainlets)
    else:
        response = api.deploy_chain(chain_name, chainlets)

    return ChainDeploymentHandle(
        chain_id=response["chain_id"],
        chain_deployment_id=response["chain_deployment_id"],
    )


def get_model_versions(api: BasetenApi, model_name: ModelName) -> Tuple[str, List]:
    query_result = api.get_model(model_name.value)["model"]
    return (query_result["id"], query_result["versions"])


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


def get_prod_version_from_versions(versions: List[dict]) -> Optional[dict]:
    # Loop over versions instead of using the primary_version field because
    # primary_version is set to the development version ID if no published
    # models exist.
    for version in versions:
        if version["is_primary"]:
            return version
    return None


def archive_truss(truss_handle: TrussHandle) -> IO:
    """
    Archive a TrussHandle into a tar file.

    Args:
        truss_handle: TrussHandle to archive

    Returns:
        A file-like object containing the tar file
    """
    truss_dir = truss_handle._spec.truss_dir
    ignore_patterns = []

    # check for a truss_ignore file and read the ignore patterns if it exists
    truss_ignore_file = truss_dir / ".truss_ignore"
    if truss_ignore_file.exists():
        ignore_patterns = load_trussignore_patterns(truss_ignore_file=truss_ignore_file)
    else:
        ignore_patterns = load_trussignore_patterns()

    try:
        temp_file = create_tar_with_progress_bar(truss_dir, ignore_patterns)
    except PermissionError:
        # workaround for Windows bug with Tempfile that causes PermissionErrors
        temp_file = create_tar_with_progress_bar(
            truss_dir, ignore_patterns, delete=False
        )
    temp_file.file.seek(0)
    return temp_file


def upload_truss(api: BasetenApi, serialize_file: IO) -> str:
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
        serialize_file.name, s3_bucket, s3_key, temp_credentials_s3_upload
    )
    return s3_key


def create_truss_service(
    api: BasetenApi,
    model_name: str,
    s3_key: str,
    config: str,
    semver_bump: str = "MINOR",
    is_trusted: bool = False,
    promote: bool = False,
    preserve_previous_prod_deployment: bool = False,
    is_draft: Optional[bool] = False,
    model_id: Optional[str] = None,
    deployment_name: Optional[str] = None,
    origin: Optional[b10_types.ModelOrigin] = None,
) -> Tuple[str, str]:
    """
    Create a model in the Baseten remote.

    Args:
        api: BasetenApi instance
        model_name: Name of the model to create
        s3_key: S3 key of the uploaded TrussHandle
        config: Base64 encoded JSON string of the Truss config
        semver_bump: Semver bump type, defaults to "MINOR"
        is_trusted: Whether the model is trusted, defaults to False
        promote: Whether to promote the model after deploy, defaults to False
        preserve_previous_prod_deployment: Wheter to scale old production deployment to zero
        deployment_name: Name to apply to the created deployment. Not applied to development model

    Returns:
        A tuple of the model ID and version ID
    """
    if is_draft:
        model_version_json = api.create_development_model_from_truss(
            model_name,
            s3_key,
            config,
            f"truss=={truss.version()}",
            is_trusted,
            origin=origin,
        )

        return (model_version_json["id"], model_version_json["version_id"])

    if model_id is None:
        model_version_json = api.create_model_from_truss(
            model_name=model_name,
            s3_key=s3_key,
            config=config,
            semver_bump=semver_bump,
            client_version=f"truss=={truss.version()}",
            is_trusted=is_trusted,
            deployment_name=deployment_name,
            origin=origin,
        )
        return (model_version_json["id"], model_version_json["version_id"])

    # Case where there is a model id already, create another version
    model_version_json = api.create_model_version_from_truss(
        model_id=model_id,
        s3_key=s3_key,
        config=config,
        semver_bump=semver_bump,
        client_version=f"truss=={truss.version()}",
        is_trusted=is_trusted,
        promote=promote,
        preserve_previous_prod_deployment=preserve_previous_prod_deployment,
        deployment_name=deployment_name,
    )
    model_version_id = model_version_json["id"]
    return (model_id, model_version_id)
