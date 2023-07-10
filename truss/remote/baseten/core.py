import logging
from typing import IO, Optional, Tuple

import truss
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.truss_handle import TrussHandle

logger = logging.getLogger(__name__)


def exists_model(api: BasetenApi, model_name: str) -> bool:
    """
    Check if a model with the given name exists in the Baseten remote.

    Args:
        api: BasetenApi instance
        model_name: Name of the model to check for existence

    Returns:
        True if the model exists, False otherwise
    """
    models = api.models()
    for model in models["models"]:
        if model["name"] == model_name:
            return True
    return False


def archive_truss(b10_truss: TrussHandle) -> IO:
    """
    Archive a TrussHandle into a tar file.

    Args:
        b10_truss: TrussHandle to archive

    Returns:
        A file-like object containing the tar file
    """
    try:
        truss_dir = b10_truss._spec.truss_dir
        temp_file = create_tar_with_progress_bar(truss_dir)
    except PermissionError:
        # Windows bug with Tempfile causes PermissionErrors
        temp_file = create_tar_with_progress_bar(truss_dir, delete=False)
    temp_file.file.seek(0)
    return temp_file


def upload_model(api: BasetenApi, serialize_file: IO) -> str:
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


def create_model(
    api: BasetenApi,
    model_name: str,
    s3_key: str,
    config: str,
    semver_bump: Optional[str] = "MINOR",
    is_trusted: Optional[bool] = False,
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
    model_version_json = api.create_model_from_truss(
        model_name,
        s3_key,
        config,
        semver_bump,
        f"truss=={truss.version()}",
        is_trusted,
    )

    return (model_version_json["id"], model_version_json["version_id"])
