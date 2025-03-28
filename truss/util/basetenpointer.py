"""This file contains the utils to create a basetenpointer from a huggingface repo, which can be resolved at runtime."""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from huggingface_hub import hf_api, hf_hub_url
from pydantic import BaseModel, RootModel

if TYPE_CHECKING:
    from truss.base.truss_config import ModelCacheV2


class Resolution(BaseModel):
    url: str
    expiration_timestamp: int


class BasetenPointer(BaseModel):
    resolution: Optional[Resolution] = None
    uid: str
    file_name: str
    hashtype: str
    hash: str
    size: int


class BasetenPointerList(RootModel):
    root: list[BasetenPointer]


def get_hf_metadata(api: "hf_api.HfApi", repo: str, revision: str, file: str):
    url = hf_hub_url(repo_id=repo, revision=revision, filename=file)
    meta = api.get_hf_file_metadata(url=url)
    return {"etag": meta.etag, "location": meta.location, "size": meta.size, "url": url}


def metadata_hf_repo(repo: str, revision: str) -> dict[str, dict]:
    """Lists all files, gathers metadata without downloading, just using the Hugging Face API.
    Example:
    [{'.gitattributes': HfFileMetadata(
    commit_hash='07163b72af1488142a360786df853f237b1a3ca1',
    etag='a6344aac8c09253b3b630fb776ae94478aa0275b',
    location='https://huggingface.co/intfloat/e5-mistral-7b-instruct/resolve/main/.gitattributes',
    url='https://huggingface.co/intfloat/e5-mistral-7b-instruct/resolve/main/.gitattributes',
    size=1519)]
    """
    api = hf_api.HfApi()
    files: list[str] = api.list_repo_files(repo_id=repo, revision=revision)

    hf_files_meta = {file: get_hf_metadata(api, repo, revision, file) for file in files}

    return hf_files_meta


def model_cache_hf_to_b10ptr(cache: "ModelCacheV2") -> BasetenPointerList:
    """
    Convert a ModelCache object to a BasetenPointer object.
    """
    assert cache is not None, "ModelCache cannot be None"

    basetenpointers: list[BasetenPointer] = []
    # validate all models have a valid revision:
    for model in cache.models:
        if model.revision is None or len(model.revision) <= 6:
            raise ValueError(
                "requiring to set the model cache revision to a valid commit sha."
                "e.g. ModelRepo(repo_id='Qwen/QwQ-32B', revision='976055f8c83f394f35dbd3ab09a285a984907bd0')"
            )
        assert model.revision is not None, (
            "ModelCache requires a valid revision for each model."
        )

        # get meta
        metadata_hf_repo_list = metadata_hf_repo(
            repo=model.repo_id, revision=model.revision
        )
        # convert the metadata to b10 pointer format
        b10_pointer_list = [
            BasetenPointer(
                uid=f"{model.repo_id}:{model.revision}:{filename}",
                file_name=(Path(model.runtime_path) / filename).as_posix(),
                hashtype="etag",
                hash=content["etag"],
                size=content["size"],
                resolution=Resolution(
                    url=content["url"],
                    # set 20 years from now for expiration, since Huggingface
                    expiration_timestamp=int(
                        time.time() + 50 * 365 * 24 * 60 * 60  # 20 years in seconds
                    ),
                ),
            )
            for filename, content in metadata_hf_repo_list.items()
        ]
        basetenpointers.extend(b10_pointer_list)

    return BasetenPointerList(root=basetenpointers)
