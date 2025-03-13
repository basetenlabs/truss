import base64
import contextlib
import json
import os
from typing import TYPE_CHECKING, Optional, Type

import boto3
from boto3.s3.transfer import TransferConfig

from truss.util.env_vars import override_env_vars

if TYPE_CHECKING:
    from rich import progress


def base64_encoded_json_str(obj):
    return base64.b64encode(str.encode(json.dumps(obj))).decode("utf-8")


def multipart_upload_boto3(
    file_path,
    bucket_name: str,
    key: str,
    credentials: dict,
    progress_bar: Optional[Type["progress.Progress"]] = None,
) -> None:
    # In the CLI flow, ignore any local ~/.aws/config files,
    # which can interfere with uploading the Truss to S3.
    with override_env_vars({"AWS_CONFIG_FILE": ""}):
        s3_resource = boto3.resource("s3", **credentials)
        filesize = os.stat(file_path).st_size

        progress_context = (
            progress_bar(transient=True) if progress_bar else contextlib.nullcontext()
        )
        task_id = (
            progress_context.add_task("[cyan]Uploading Truss", total=filesize)
            if not isinstance(progress_context, contextlib.nullcontext)
            else None
        )

        def callback(bytes_transferred):
            if progress_bar:
                progress_context.update(task_id, advance=bytes_transferred)

        with progress_context:
            s3_resource.Object(bucket_name, key).upload_file(
                file_path,
                Config=TransferConfig(max_concurrency=10, use_threads=True),
                Callback=callback,
            )
