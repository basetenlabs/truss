import datetime
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from google.cloud import storage
from huggingface_hub import hf_hub_download

B10CP_PATH_TRUSS_ENV_VAR_NAME = "B10CP_PATH_TRUSS"

GCS_CREDENTIALS = "/app/data/service_account.json"
S3_CREDENTIALS = "/app/data/s3_credentials.json"


def _b10cp_path() -> Optional[str]:
    return os.environ.get(B10CP_PATH_TRUSS_ENV_VAR_NAME)


def _download_from_url_using_b10cp(b10cp_path: str, url: str, download_to: Path):
    return subprocess.Popen(
        [
            b10cp_path,
            "-source",
            url,  # Add quotes to work with any special characters.
            "-target",
            str(download_to),
        ]
    )


@dataclass
class AWSCredentials:
    access_key_id: str
    secret_access_key: str
    region: str
    session_token: Optional[str]


def parse_s3_credentials_file(key_file_path: str) -> AWSCredentials:
    # open the json file
    with open(key_file_path, "r") as f:
        data = json.load(f)

    # validate the data
    if (
        "aws_access_key_id" not in data
        or "aws_secret_access_key" not in data
        or "aws_region" not in data
    ):
        raise ValueError("Invalid AWS credentials file")

    # create an AWS Service Account object
    aws_sa = AWSCredentials(
        access_key_id=data["aws_access_key_id"],
        secret_access_key=data["aws_secret_access_key"],
        region=data["aws_region"],
        session_token=data.get("aws_session_token", None),
    )

    return aws_sa


def split_path(path, prefix="gs://"):
    # Remove the 'gs://' prefix
    path = path.replace(prefix, "")

    # Split on the first slash
    parts = path.split("/", 1)

    bucket_name = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    return bucket_name, path


class RepositoryFile(ABC):
    def __init__(self, repo_name, file_name, revision_name):
        self.repo_name = repo_name
        self.file_name = file_name
        self.revision_name = revision_name

    @staticmethod
    def from_file(
        new_repo_name: str, new_file_name: str, new_revision_name: str
    ) -> "RepositoryFile":
        repository_class: Type["RepositoryFile"]
        if new_repo_name.startswith("gs://"):
            repository_class = GCSFile
        elif new_repo_name.startswith("s3://"):
            repository_class = S3File
        else:
            repository_class = HuggingFaceFile
        return repository_class(new_repo_name, new_file_name, new_revision_name)

    @abstractmethod
    def download_to_cache(self):
        pass


class HuggingFaceFile(RepositoryFile):
    def download_to_cache(self):
        secret_path = Path("/etc/secrets/hf-access-token")
        secret = secret_path.read_text().strip() if secret_path.exists() else None
        try:
            hf_hub_download(
                self.repo_name,
                self.file_name,
                revision=self.revision_name,
                token=secret,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Hugging Face repository not found (and no valid secret found for possibly private repository)."
            )


class GCSFile(RepositoryFile):
    def download_to_cache(self):
        # Create GCS Client
        bucket_name, _ = split_path(repo_name, prefix="gs://")

        is_private = os.path.exists(GCS_CREDENTIALS)
        print(is_private)
        if is_private:
            print("loading...")
            client = storage.Client.from_service_account_json(GCS_CREDENTIALS)
        else:
            client = storage.Client.create_anonymous_client()

        bucket = client.bucket(bucket_name)

        # Cache file
        cache_dir = Path(f"/app/model_cache/{bucket_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        dst_file = cache_dir / self.file_name
        if not dst_file.parent.exists():
            dst_file.parent.mkdir(parents=True)

        blob = bucket.blob(self.file_name)

        if not blob.exists(client):
            raise RuntimeError(f"File not found on GCS bucket: {blob.name}")

        if is_private:
            url = blob.generate_signed_url(
                version="v4", expiration=datetime.timedelta(minutes=15), method="GET"
            )
        else:
            base_url = "https://storage.googleapis.com"
            url = f"{base_url}/{bucket_name}/{blob.name}"

        download_file_using_b10cp(url, dst_file, self.file_name)


class S3File(RepositoryFile):
    def download_to_cache(self):
        # Create S3 Client
        bucket_name, _ = split_path(repo_name, prefix="s3://")

        if os.path.exists(S3_CREDENTIALS):
            s3_credentials = parse_s3_credentials_file(S3_CREDENTIALS)
            client = boto3.client(
                "s3",
                aws_access_key_id=s3_credentials.access_key_id,
                aws_secret_access_key=s3_credentials.secret_access_key,
                region_name=s3_credentials.region,
                aws_session_token=s3_credentials.session_token,
                config=Config(signature_version="s3v4"),
            )
        else:
            client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        # Cache file
        cache_dir = Path(f"/app/model_cache/{bucket_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        dst_file = cache_dir / self.file_name
        if not dst_file.parent.exists():
            dst_file.parent.mkdir(parents=True)

        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": file_name},
                ExpiresIn=3600,
            )
        except NoCredentialsError as nce:
            raise RuntimeError(
                f"No AWS credentials found\nOriginal exception: {str(nce)}"
            )
        except ClientError as ce:
            raise RuntimeError(
                f"Client error when accessing the S3 bucket (check your credentials): {str(ce)}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"File not found on S3 bucket: {file_name}\nOriginal exception: {str(exc)}"
            )

        download_file_using_b10cp(url, dst_file, self.file_name)


def download_file_using_b10cp(url, dst_file, file_name):
    try:
        proc = _download_from_url_using_b10cp(_b10cp_path(), url, dst_file)
        proc.wait()

    except FileNotFoundError as file_error:
        raise RuntimeError(f"Failure due to file ({file_name}) not found: {file_error}")


def download_file(repo_name, file_name, revision_name=None):
    file = RepositoryFile.from_file(repo_name, file_name, revision_name)
    file.download_to_cache()


if __name__ == "__main__":
    file_path = Path.home() / ".cache/huggingface/hub/version.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.is_file():
        file_path.write_text("1")

    file_name = sys.argv[1]
    repo_name = sys.argv[2]
    revision_name = sys.argv[3] if len(sys.argv) >= 4 else None

    download_file(repo_name, file_name, revision_name)
