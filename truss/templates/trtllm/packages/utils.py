import socket
from pathlib import Path

import tritonclient.grpc as grpcclient
from constants import (
    GRPC_SERVICE_PORT,
    HTTP_SERVICE_PORT,
    TENSORRT_LLM_MODEL_REPOSITORY_PATH,
)
from huggingface_hub import snapshot_download
from tritonclient.utils import np_to_triton_dtype


def move_all_files(src: Path, dest: Path) -> None:
    """
    Moves all files from `src` to `dest` recursively.
    """
    print(f"Moving from {src} to {dest}")
    for item in src.iterdir():
        dest_item = dest / item.name
        if item.is_dir():
            dest_item.mkdir(parents=True, exist_ok=True)
            move_all_files(item, dest_item)
        else:
            item.rename(dest_item)


def prepare_model_repository(data_dir: Path) -> None:
    # Ensure the destination directory exists
    dest_dir = TENSORRT_LLM_MODEL_REPOSITORY_PATH / "tensorrt_llm" / "1"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Ensure empty version directory for `ensemble` model exists
    ensemble_dir = TENSORRT_LLM_MODEL_REPOSITORY_PATH / "ensemble" / "1"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Move all files and directories from data_dir to dest_dir
    move_all_files(data_dir, dest_dir)


def prepare_grpc_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def download_engine(engine_repository: str, fp: Path, auth_token=None):
    """
    Downloads the specified engine from Hugging Face Hub.
    """
    snapshot_download(
        engine_repository,
        local_dir=fp,
        local_dir_use_symlinks=False,
        max_workers=4,
        **({"use_auth_token": auth_token} if auth_token is not None else {}),
    )


def server_loaded():
    def port_is_available(port):
        available = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                available = True
            except OSError:
                pass
        return available

    return not port_is_available(GRPC_SERVICE_PORT) or not port_is_available(
        HTTP_SERVICE_PORT
    )
