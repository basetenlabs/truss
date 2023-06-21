import base64
import json
import os

import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm


def base64_encoded_json_str(obj):
    return base64.b64encode(str.encode(json.dumps(obj))).decode("utf-8")


def multipart_upload_boto3(file_path, bucket_name, key, credentials):
    s3_resource = boto3.resource("s3", **credentials)
    filesize = os.stat(file_path).st_size

    with tqdm(
        total=filesize,
        desc="Upload",
        unit="B",
        unit_scale=True,
    ) as pbar:
        s3_resource.Object(bucket_name, key).upload_file(
            file_path,
            Config=TransferConfig(
                max_concurrency=10,
                use_threads=True,
            ),
            Callback=pbar.update,
        )
