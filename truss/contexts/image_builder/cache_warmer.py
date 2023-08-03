import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_file(repo_name, file_name, revision_name=None):
    secret = None
    secret_path = Path("/etc/secrets/hf_access_token")

    if secret_path.exists():
        secret = secret_path.read_text().strip()
    try:
        hf_hub_download(repo_name, file_name, revision=revision_name, token=secret)
    except FileNotFoundError:
        raise RuntimeError(
            "Hugging Face repository not found (and no valid secret found for possibly private repository)."
        )


if __name__ == "__main__":
    # TODO(varun): might make sense to move this check + write to a separate `prepare_cache.py` script
    file_path = Path.home() / ".cache/huggingface/hub/version.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.is_file():
        file_path.write_text("1")

    file_name = sys.argv[1]
    repo_name = sys.argv[2]
    revision_name = sys.argv[3] if len(sys.argv) >= 4 else None

    download_file(repo_name, file_name, revision_name)
