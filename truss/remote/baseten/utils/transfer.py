import base64
import json
import os

import boto3
from boto3.s3.transfer import TransferConfig
from rich.progress import Progress


def base64_encoded_json_str(obj):
    return base64.b64encode(str.encode(json.dumps(obj))).decode("utf-8")


def multipart_upload_boto3(file_path, bucket_name, key, credentials):
    s3_resource = boto3.resource("s3", **credentials)
    filesize = os.stat(file_path).st_size

    # Create a new progress bar
    progress = Progress()

    # Add a new task to the progress bar
    task_id = progress.add_task("[cyan]Uploading...", total=filesize)

    with progress:

        def callback(bytes_transferred):
            # Update the progress bar
            progress.update(task_id, advance=bytes_transferred)

        s3_resource.Object(bucket_name, key).upload_file(
            file_path,
            Config=TransferConfig(
                max_concurrency=10,
                use_threads=True,
            ),
            Callback=callback,
        )
