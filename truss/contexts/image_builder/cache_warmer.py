import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


def download_file(repo_name, file_name, revision_name=None):
    try:
        hf_hub_download(repo_name, file_name, revision=revision_name)
    except RepositoryNotFoundError:
        try:
            with open("/etc/secrets/hf_secret", "r") as secretFile:
                secret = secretFile.read().strip()

            # Retry after setting the secret.
            hf_hub_download(repo_name, file_name, revision=revision_name, token=secret)
        except FileNotFoundError:
            raise RuntimeError(
                "Hugging Face repository not found (and no valid secret found for possibly private repository)."
            )


if __name__ == "__main__":
    file_name = sys.argv[1]
    repo_name = sys.argv[2]
    revision_name = sys.argv[3] if len(sys.argv) >= 4 else None

    download_file(repo_name, file_name, revision_name)
