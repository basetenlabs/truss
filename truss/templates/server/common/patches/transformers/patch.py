"""
Patch for Huggingface/Transformer:

The Transformer's library currently performs serial downloads of weights, which can
be time-consuming when there are multiple weights to be fetched. This patch aims to
parallelize the downloads using joblib, allowing the downloads to be spread across
multiple cores and improving overall performance.
"""
import os

from joblib import Parallel, delayed
from transformers.utils import hub


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


def patch():
    """
    Patching Huggingface downloads to make them parallelized.
    """
    hub.get_checkpoint_shard_files = get_checkpoint_shard_files
