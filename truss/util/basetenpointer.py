"""This file contains the utils to create a basetenpointer from a huggingface repo, which can be resolved at runtime."""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import requests
from huggingface_hub import hf_api, hf_hub_url
from huggingface_hub.utils import filter_repo_objects
from pydantic import BaseModel

if TYPE_CHECKING:
    from truss.base.truss_config import ModelCache


# copied from: https://github.com/basetenlabs/baseten/blob/caeba66cd544a5152bb6a018d6ac2871814f327b/baseten_shared/baseten_shared/lms/types.py#L13
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


class BasetenPointerList(BaseModel):
    pointers: list[BasetenPointer]


def get_hf_metadata(api: "hf_api.HfApi", repo: str, revision: str, file: str):
    url = hf_hub_url(repo_id=repo, revision=revision, filename=file)
    meta = api.get_hf_file_metadata(url=url)
    return {"etag": meta.etag, "location": meta.location, "size": meta.size, "url": url}


def filter_repo_files(
    files: list[str],
    allow_patterns: Optional[list[str]],
    ignore_patterns: Optional[list[str]],
) -> list[str]:
    return list(
        filter_repo_objects(
            items=files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
        )
    )


def metadata_hf_repo(
    repo: str,
    revision: str,
    allow_patterns: Optional[list[str]] = None,
    ignore_patterns: Optional[list[str]] = None,
) -> dict[str, dict]:
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
    files = filter_repo_files(
        files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
    )

    hf_files_meta = {file: get_hf_metadata(api, repo, revision, file) for file in files}

    return hf_files_meta


def model_cache_hf_to_b10ptr(cache: "ModelCache") -> BasetenPointerList:
    """
    Convert a ModelCache object to a BasetenPointer object.
    """
    assert cache.is_v2, "ModelCache is not v2"

    basetenpointers: list[BasetenPointer] = []

    for model in cache.models:
        assert model.revision is not None, "ModelCache is not v2, revision is None"
        exception = None
        for _ in range(3):
            try:
                metadata_hf_repo_list = metadata_hf_repo(
                    repo=model.repo_id,
                    revision=model.revision,
                    allow_patterns=model.allow_patterns,
                    ignore_patterns=model.ignore_patterns,
                )
                break
            except requests.exceptions.ReadTimeout as e:
                # this is expected, sometimes huggingface hub times out
                print("ReadTimeout Error: ", e)
                time.sleep(5)
                exception = e
            except Exception as e:
                raise e
        else:
            # if we get here, we have exhausted the retries
            assert exception is not None, "ReadTimeout Error: " + str(exception)
            raise exception
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
                    expiration_timestamp=int(
                        4044816725  # 90 years in the future, hf does not expire. needs to be static, to have cache hits.
                    ),
                ),
            )
            for filename, content in metadata_hf_repo_list.items()
        ]
        basetenpointers.extend(b10_pointer_list)

    return BasetenPointerList(pointers=basetenpointers)
