import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from google.cloud import storage
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
B10CP_PATH_TRUSS_ENV_VAR_NAME = "B10CP_PATH_TRUSS"


def _b10cp_path() -> Optional[str]:
    return os.environ.get(B10CP_PATH_TRUSS_ENV_VAR_NAME)


def _download_from_url_using_b10cp(
    b10cp_path: str,
    url: str,
    download_to: Path,
):
    return subprocess.Popen(
        [
            b10cp_path,
            "-source",
            url,  # Add quotes to work with any special characters.
            "-target",
            str(download_to),
        ]
    )


def split_path(path, prefix="gs://"):
    # Remove the 'gs://' prefix
    path = path.replace(prefix, "")

    # Split on the first slash
    parts = path.split("/", 1)

    bucket_name = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    return bucket_name, path


def parse_s3_service_account_file(file_path):
    # open the json file
    with open(file_path, "r") as f:
        data = json.load(f)

    # validate the data
    if "aws_access_key_id" not in data or "aws_secret_access_key" not in data:
        raise ValueError("Invalid AWS credentials file")

    # parse the data
    aws_access_key_id = data["aws_access_key_id"]
    aws_secret_access_key = data["aws_secret_access_key"]
    aws_region = data["aws_region"]

    return aws_access_key_id, aws_secret_access_key, aws_region


def download_file(
    repo_name, file_name, revision_name=None, key_file="/app/data/service_account.json"
):
    # Check if repo_name starts with "gs://"
    if repo_name.startswith(("gs://", "s3://")):
        prefix = repo_name[:5]

        # Create directory if not exist
        bucket_name, _ = split_path(repo_name, prefix=prefix)
        repo_name = repo_name.replace(prefix, "")
        cache_dir = Path(f"/app/hf_cache/{bucket_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        if prefix == "gs://":
            # Connect to GCS storage
            storage_client = storage.Client.from_service_account_json(key_file)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            dst_file = Path(f"{cache_dir}/{file_name}")
            if not dst_file.parent.exists():
                dst_file.parent.mkdir(parents=True)

            if not blob.exists(storage_client):
                raise RuntimeError(f"File not found on GCS bucket: {blob.name}")

            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=15),
                method="GET",
            )
            try:
                proc = _download_from_url_using_b10cp(_b10cp_path(), url, dst_file)
                proc.wait()
            except Exception as e:
                raise RuntimeError(f"Failure downloading file from GCS: {e}")
        elif prefix == "s3://":
            (
                AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY,
                AWS_REGION,
            ) = parse_s3_service_account_file(key_file)
            client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
                config=Config(signature_version="s3v4"),
            )
            bucket_name, _ = split_path(bucket_name, prefix="s3://")

            dst_file = Path(f"{cache_dir}/{file_name}")
            if not dst_file.parent.exists():
                dst_file.parent.mkdir(parents=True)

            try:
                url = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": file_name},
                    ExpiresIn=3600,
                )
            except Exception:
                raise RuntimeError(f"File not found on S3 bucket: {file_name}")

            try:
                proc = _download_from_url_using_b10cp(_b10cp_path(), url, dst_file)
                proc.wait()
            except Exception as e:
                raise RuntimeError(f"Failure downloading file from S3: {e}")
    else:
        secret_path = Path("/etc/secrets/hf-access-token")
        secret = secret_path.read_text().strip() if secret_path.exists() else None
        try:
            hf_hub_download(
                repo_name,
                file_name,
                revision=revision_name,
                token=secret,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Hugging Face repository not found (and no valid secret found for possibly private repository)."
            )


if __name__ == "__main__":
    file_path = Path.home() / ".cache/huggingface/hub/version.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.is_file():
        file_path.write_text("1")

    file_name = sys.argv[1]
    repo_name = sys.argv[2]
    revision_name = sys.argv[3] if len(sys.argv) >= 4 else None

    download_file(repo_name, file_name, revision_name)
