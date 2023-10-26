import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore import UNSIGNED
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


def split_path(path, prefix="gs://"):
    # Remove the 'gs://' prefix
    path = path.replace(prefix, "")

    # Split on the first slash
    parts = path.split("/", 1)

    bucket_name = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    return bucket_name, path


class RepositoryFile:
    def __init__(self, repo_name, file_name, revision_name):
        self.repo_name = repo_name
        self.file_name = file_name
        self.revision_name = revision_name
        self.client = None
        self.bucket_name = None
        self.bucket = None

    @staticmethod
    def create(repo_name, file_name, revision_name):
        if repo_name.startswith("gs://"):
            repository_class = GCSFile
        elif repo_name.startswith("s3://"):
            repository_class = S3File
        else:
            repository_class = HuggingFaceFile
        return repository_class(repo_name, file_name, revision_name)

    def connect(self, key_file=None):
        # connect to the correct repo and set up accessors
        pass

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
    def connect(self, key_file="/app/data/service_account.json"):
        self.bucket_name, _ = split_path(repo_name, prefix="gs://")

        # Connect to GCS storage
        self.is_private = os.path.exists(key_file)
        if self.is_private:
            self.client = storage.Client.from_service_account_json(key_file)
        else:
            self.client = storage.Client.create_anonymous_client()

        self.bucket = self.client.bucket(self.bucket_name)

    def download_to_cache(self):
        cache_dir = Path(f"/app/hf_cache/{self.bucket_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        dst_file = Path(f"{cache_dir}/{self.file_name}")
        if not dst_file.parent.exists():
            dst_file.parent.mkdir(parents=True)

        blob = self.bucket.blob(self.file_name)

        if not blob.exists(self.client):
            raise RuntimeError(f"File not found on GCS bucket: {blob.name}")

        if self.is_private:
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=15),
                method="GET",
            )
        else:
            base_url = "https://storage.googleapis.com"
            url = f"{base_url}/{self.bucket_name}/{blob.name}"
        try:
            proc = _download_from_url_using_b10cp(_b10cp_path(), url, dst_file)
            proc.wait()
        except Exception as e:
            raise RuntimeError(f"Failure downloading file from GCS: {e}")


class S3File(RepositoryFile):
    def connect(self, key_file="/app/data/s3_credentials.json"):
        self.bucket_name, _ = split_path(repo_name, prefix="s3://")

        if os.path.exists(key_file):
            (
                AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY,
                AWS_REGION,
            ) = parse_s3_service_account_file(key_file)
            self.client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
                config=Config(signature_version="s3v4"),
            )
        else:
            self.client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def download_to_cache(self):
        cache_dir = Path(f"/app/hf_cache/{self.bucket_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        dst_file = Path(f"{cache_dir}/{self.file_name}")
        if not dst_file.parent.exists():
            dst_file.parent.mkdir(parents=True)

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": file_name},
                ExpiresIn=3600,
            )
        except Exception:
            raise RuntimeError(f"File not found on S3 bucket: {file_name}")

        try:
            proc = _download_from_url_using_b10cp(_b10cp_path(), url, dst_file)
            proc.wait()
        except Exception as e:
            raise RuntimeError(f"Failure downloading file from S3: {e}")


def download_file(repo_name, file_name, revision_name=None):
    file = RepositoryFile.create(repo_name, file_name, revision_name)
    file.connect()
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
