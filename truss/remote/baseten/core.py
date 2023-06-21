import logging
from typing import IO, Optional, Tuple

import truss
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.truss_handle import TrussHandle

logger = logging.getLogger(__name__)


def exists_model(api: BasetenApi, model_name: str) -> bool:
    models = api.models()
    model_id_by_name = {model["name"]: model["id"] for model in models["models"]}
    return model_name in model_id_by_name


def archive_truss(b10_truss: TrussHandle) -> IO:
    try:
        truss_dir = b10_truss._spec.truss_dir
        temp_file = create_tar_with_progress_bar(truss_dir)
    except PermissionError:
        # Windows bug with Tempfile causes PermissionErrors
        temp_file = create_tar_with_progress_bar(truss_dir, delete=False)
    temp_file.file.seek(0)
    return temp_file


def upload_model(api: BasetenApi, serialize_file: IO) -> str:
    temp_credentials_s3_upload = api.model_s3_upload_credentials()
    s3_key = temp_credentials_s3_upload.pop("s3_key")
    s3_bucket = temp_credentials_s3_upload.pop("s3_bucket")
    logger.info("🚀 Uploading model to Baseten 🚀")
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
    external_model_version_id: Optional[str] = None,
) -> Tuple[str, str]:
    model_version_json = api.create_model_from_truss(
        model_name,
        s3_key,
        config,
        semver_bump,
        f"truss=={truss.version()}",
        is_trusted,
        external_model_version_id,
    )

    return (model_version_json["id"], model_version_json["version_id"])
