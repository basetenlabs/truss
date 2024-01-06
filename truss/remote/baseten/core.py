import logging
from typing import IO, List, Optional, Tuple

import truss
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.truss_handle import TrussHandle

logger = logging.getLogger(__name__)


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
        b10_truss: TrussHandle to archive

    Returns:
        A file-like object containing the tar file
    """
    try:
        truss_dir = truss_handle._spec.truss_dir
        temp_file = create_tar_with_progress_bar(truss_dir)
    except PermissionError:
        # Windows bug with Tempfile causes PermissionErrors
        temp_file = create_tar_with_progress_bar(truss_dir, delete=False)
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
    is_draft: Optional[bool] = False,
    model_id: Optional[str] = None,
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
    )
    model_version_id = model_version_json["id"]
    return (model_id, model_version_id)
