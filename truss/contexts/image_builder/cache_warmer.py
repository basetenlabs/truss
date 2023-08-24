import os
import sys
from pathlib import Path

from google.cloud import storage
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def download_file(
    repo_name, file_name, revision_name=None, key_file="/app/data/service_account.json"
):
    # Check if repo_name starts with "gs://"
    if "gs://" in repo_name:
        # Create directory if not exist
        repo_name = repo_name.replace("gs://", "")
        cache_dir = Path(f"/app/hf_cache/{repo_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Connect to GCS storage
        try:
            storage_client = storage.Client.from_service_account_json(key_file)
            bucket = storage_client.bucket(repo_name)
            blob = bucket.blob(file_name)
            dst_file = Path(f"{cache_dir}/{file_name}")
            if not dst_file.parent.exists():
                dst_file.parent.mkdir(parents=True)
            # Download the blob to a file
            blob.download_to_filename(dst_file)
        except Exception as e:
            raise RuntimeError(f"Failure downloading file from GCS: {e}")
    else:
        secret_path = Path("/etc/secrets/hf_access_token")
        secret = secret_path.read_text().strip() if secret_path.exists() else None
        try:
            hf_hub_download(repo_name, file_name, revision=revision_name, token=secret)
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
