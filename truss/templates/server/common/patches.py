import logging
import os
from typing import Union

from joblib import Parallel, delayed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _download(url: str, root: str, in_memory: bool) -> Union[bytes, str]:
    import hashlib
    import urllib
    import warnings

    from tqdm import tqdm

    os.makedirs(root, exist_ok=True)
    print("DOWNLOADING")
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length"))
            if source.info().get("Content-Length") is not None
            else None,
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target


def whisper_patch():
    import whisper

    whisper._download = _download


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    _commit_hash=None,
):
    import json

    from huggingface_hub.utils import EntryNotFoundError
    from requests.exceptions import HTTPError
    from transformers.utils.hub import HUGGINGFACE_CO_RESOLVE_ENDPOINT, cached_file

    if not os.path.isfile(index_filename):
        raise ValueError(
            f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [
            os.path.join(pretrained_model_name_or_path, subfolder, f)
            for f in shard_filenames
        ]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []

    # Function to load file from url
    def load_file_from_url(shard_filename):
        try:
            # Load from URL
            return cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

    # Use joblib to download in parallel
    cached_filenames = Parallel(n_jobs=-1)(
        delayed(load_file_from_url)(shard_filename)
        for shard_filename in shard_filenames
    )

    # Filter out None results in case of errors
    cached_filenames = [f for f in cached_filenames if f is not None]

    return cached_filenames, sharded_metadata


def huggingface_patch():
    """
    Patching Huggingface downloads to make them parallelized.
    """
    from transformers.utils import hub

    hub.get_checkpoint_shard_files = get_checkpoint_shard_files


def apply_patches(enabled: bool, requirements: list):
    """
    Apply patches to certain functions. The patches are contained in the PATCHES list.
    If a patch cannot be applied, it logs the name of the function and the exception details.
    """
    PATCHES = {
        "transformers": huggingface_patch,
        "whisper": whisper_patch,
    }
    if not enabled:
        return
    for requirement in requirements:
        # We iterate over patches so that we can check if the patch_name exists as a substring
        # of the requirement such as for git url requirements.
        for patch_name in PATCHES:
            if patch_name in requirement:
                try:
                    PATCHES[patch_name]()
                except Exception as e:
                    logger.debug(
                        f"{patch_name} patch could not be applied. Exception: {str(e)}"
                    )
