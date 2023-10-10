import logging
from typing import IO, Optional, Tuple

import truss
from truss.remote.baseten.api import BasetenApi
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
    models = api.models()
    for model in models["models"]:
        if model["name"] == model_name:
            return model["id"]
    return None


def get_model_versions_info(api: BasetenApi, model_name) -> Tuple[str, dict]:
    query_result = api.get_model(model_name)["model_version"]["oracle"]
    return (query_result["id"], query_result["versions"])


def get_dev_version_info(api: BasetenApi, model_name: str) -> dict:
    model = api.get_model(model_name)
    versions = model["model_version"]["oracle"]["versions"]
    for version in versions:
        if version["is_draft"] is True:
            return version
    raise ValueError(f"No development version found with model name: {model_name}")


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
    )
    model_version_id = model_version_json["id"]
    return (model_id, model_version_id)
