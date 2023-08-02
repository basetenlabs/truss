import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_file(repo_name, file_name, revision_name=None):
    secret = None
    if Path("/etc/secrets/hf_secret").exists():
        with open("/etc/secrets/hf_secret", "r") as secretFile:
            secret = secretFile.read().strip()
    try:
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
