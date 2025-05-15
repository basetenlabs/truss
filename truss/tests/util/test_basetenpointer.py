import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import requests

from truss.base.truss_config import ModelCache, ModelRepo
from truss.util.basetenpointer import model_cache_hf_to_b10ptr


def test_dolly_12b():
    ModelCached = ModelCache(
        [
            dict(
                repo_id="databricks/dolly-v2-12b",
                revision="19308160448536e378e3db21a73a751579ee7fdd",
                use_volume=True,
                volume_folder="databricks_dolly_v2_12b",
            )
        ]
    )
    for _ in range(2):
        try:
            bptr = model_cache_hf_to_b10ptr(ModelCached)
            continue
        # timeout by huggingface hub timeout error
        except requests.exceptions.ReadTimeout as e:
            # this is expected to timeout when the request takes too long
            # due to the large size of the model
            print("ReadTimeout Error: ", e)
            pytest.skip(
                "Skipping test due to ReadTimeout error from Hugging Face API, "
                "this can happen for large models like Dolly-12b"
            )
    bptr_list = bptr.pointers
    expected = [
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/.gitattributes",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:.gitattributes",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/.gitattributes",
            "hashtype": "etag",
            "hash": "c7d9f3332a950355d5a77d85000f05e6f45435ea",
            "size": 1477,
        },
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/README.md",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:README.md",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/README.md",
            "hashtype": "etag",
            "hash": "2912eb39545af0367335cff448d07214519c5eed",
            "size": 10746,
        },
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/config.json",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:config.json",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/config.json",
            "hashtype": "etag",
            "hash": "888c677eda015e2375fad52d75062d14b30ebad9",
            "size": 818,
        },
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/instruct_pipeline.py",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:instruct_pipeline.py",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/instruct_pipeline.py",
            "hashtype": "etag",
            "hash": "f8b291569e936cf104f44d003f95451bf5e1f965",
            "size": 9159,
        },
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/pytorch_model.bin",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:pytorch_model.bin",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/pytorch_model.bin",
            "hashtype": "etag",
            "hash": "19e10711310992c310c3775964c7635f4b28dd86587403e718c6d6d524a406a5",
            "size": 23834965761,
        },
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/special_tokens_map.json",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:special_tokens_map.json",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/special_tokens_map.json",
            "hashtype": "etag",
            "hash": "ecc1ee07dec13ee276fa9f1b29a1078da3280a4d",
            "size": 228,
        },
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/tokenizer.json",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:tokenizer.json",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/tokenizer.json",
            "hashtype": "etag",
            "hash": "22868c8caf99a303c1a44bfea98f20f4254fc0e5",
            "size": 2114274,
        },
        {
            "resolution": {
                "url": "https://huggingface.co/databricks/dolly-v2-12b/resolve/19308160448536e378e3db21a73a751579ee7fdd/tokenizer_config.json",
                "expiration_timestamp": 2373918212,
            },
            "uid": "databricks/dolly-v2-12b:19308160448536e378e3db21a73a751579ee7fdd:tokenizer_config.json",
            "file_name": "/app/model_cache/databricks_dolly_v2_12b/tokenizer_config.json",
            "hashtype": "etag",
            "hash": "51e564ead5d28eebc74b25d86f0a694b7c7cc618",
            "size": 449,
        },
    ]
    assert len(bptr_list) == len(expected), (
        f"Expected {len(expected)} but got {len(bptr_list)}"
    )
    for expected, actual in zip(expected, bptr_list):
        assert expected["uid"] == actual.uid, (
            f"Expected uid {expected['uid']} but got {actual.uid}"
        )
        assert expected["file_name"] == actual.file_name, (
            f"Expected file_name {expected['file_name']} but got {actual.file_name}"
        )
        assert expected["hash"] == actual.hash, (
            f"Expected hash {expected['hash']} but got {actual.hash}"
        )
        assert expected["size"] == actual.size, (
            f"Expected size {expected['size']} but got {actual.size}"
        )
        assert expected["resolution"]["url"] == actual.resolution.url, (
            f"Expected resolution url {expected['resolution']['url']} but got {actual.resolution.url}"
        )
        # 100 years or more ahead
        assert (
            actual.resolution.expiration_timestamp
            >= time.time() + 20 * 365 * 24 * 60 * 60
        ), (
            f"Expected unix expiration timestamp to be at least 20 years ahead, but got {actual.resolution.expiration_timestamp}. "
        )

    # download first file and verify size
    with TemporaryDirectory() as tmp:
        # Get the first pointer (.gitattributes)
        first_pointer = bptr_list[0]
        tmp_path = Path(tmp) / "downloaded_file"

        # Download the file
        response = requests.get(first_pointer.resolution.url)
        response.raise_for_status()

        # Save the file
        tmp_path.write_bytes(response.content)

        # Verify file size matches metadata
        actual_size = tmp_path.stat().st_size
        assert actual_size == first_pointer.size, (
            f"Downloaded file size {actual_size} does not match expected size {first_pointer.size}"
        )


def test_with_main():
    # main should be resolved to 41dec486b25746052d3335decc8f5961607418a0
    cache = ModelCache(
        [
            ModelRepo(
                repo_id="intfloat/llm-retriever-base",
                revision="main",
                ignore_patterns=["*.json", "*.txt", "*.md", "*.bin", "*.model"],
                volume_folder="mistral_demo",
                use_volume=True,
            )
        ]
    )
    b10ptr = model_cache_hf_to_b10ptr(cache)
    expected = {
        "pointers": [
            {
                "resolution": {
                    "url": "https://huggingface.co/intfloat/llm-retriever-base/resolve/41dec486b25746052d3335decc8f5961607418a0/.gitattributes",
                    "expiration_timestamp": 4044816725,
                },
                "uid": "intfloat/llm-retriever-base:main:.gitattributes",
                "file_name": "/app/model_cache/mistral_demo/.gitattributes",
                "hashtype": "etag",
                "hash": "a6344aac8c09253b3b630fb776ae94478aa0275b",
                "size": 1519,
            },
            {
                "resolution": {
                    "url": "https://huggingface.co/intfloat/llm-retriever-base/resolve/41dec486b25746052d3335decc8f5961607418a0/model.safetensors",
                    "expiration_timestamp": 4044816725,
                },
                "uid": "intfloat/llm-retriever-base:main:model.safetensors",
                "file_name": "/app/model_cache/mistral_demo/model.safetensors",
                "hashtype": "etag",
                "hash": "565dd4f1cc6318ccf07af8680c27fd935b3b56ca2684d1af58abcd4e8bf6ecfa",
                "size": 437955512,
            },
        ]
    }
    assert b10ptr.model_dump() == expected


if __name__ == "__main__":
    test_dolly_12b()
    test_with_main()
