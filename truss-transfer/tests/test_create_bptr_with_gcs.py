import json
from pathlib import Path

import pytest
import truss_transfer

MANIFEST_EXPECTED = [
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "LICENSE.txt",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-d665b7f3a12a6b763c45af10708c4e67",
        "file_name": "/app/model_cache/julien_dummy/LICENSE.txt",
        "hashtype": "md5",
        "hash": "d665b7f3a12a6b763c45af10708c4e67",
        "size": 7712,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "tokenizer.json",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-3ff6b653d22a2676f3a03bd7b5d7ff88",
        "file_name": "/app/model_cache/julien_dummy/tokenizer.json",
        "hashtype": "md5",
        "hash": "3ff6b653d22a2676f3a03bd7b5d7ff88",
        "size": 9085657,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "special_tokens_map.json",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-ff9bb51206c5b33aa9385a1a371294b6",
        "file_name": "/app/model_cache/julien_dummy/special_tokens_map.json",
        "hashtype": "md5",
        "hash": "ff9bb51206c5b33aa9385a1a371294b6",
        "size": 296,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "README.md",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-35abb0d23c1471c8d5374d6cd861b112",
        "file_name": "/app/model_cache/julien_dummy/README.md",
        "hashtype": "md5",
        "hash": "35abb0d23c1471c8d5374d6cd861b112",
        "size": 41742,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "generation_config.json",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-ed571b74f968dd1a6791128e3d484a0b",
        "file_name": "/app/model_cache/julien_dummy/generation_config.json",
        "hashtype": "md5",
        "hash": "ed571b74f968dd1a6791128e3d484a0b",
        "size": 189,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "USE_POLICY.md",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-1729addd533ca7a6e956e3f077f4a4e9",
        "file_name": "/app/model_cache/julien_dummy/USE_POLICY.md",
        "hashtype": "md5",
        "hash": "1729addd533ca7a6e956e3f077f4a4e9",
        "size": 6021,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "config.json",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-4eb556c0849743773f374eefd719b008",
        "file_name": "/app/model_cache/julien_dummy/config.json",
        "hashtype": "md5",
        "hash": "4eb556c0849743773f374eefd719b008",
        "size": 877,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "tokenizer_config.json",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-dc8ebd5b3ad223ab5b3c64d47975f23a",
        "file_name": "/app/model_cache/julien_dummy/tokenizer_config.json",
        "hashtype": "md5",
        "hash": "dc8ebd5b3ad223ab5b3c64d47975f23a",
        "size": 54528,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": ".gitattributes",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-a859f8a89685747ffd4171b870540c41",
        "file_name": "/app/model_cache/julien_dummy/.gitattributes",
        "hashtype": "md5",
        "hash": "a859f8a89685747ffd4171b870540c41",
        "size": 1519,
        "runtime_secret_name": "gcs-account",
    },
    {
        "resolution": {
            "resolution_type": "gcs",
            "path": "model.safetensors",
            "bucket_name": "llama-3-2-1b-instruct",
        },
        "uid": "gcs-4f83b78475e5582a014664915eb222d7",
        "file_name": "/app/model_cache/julien_dummy/model.safetensors",
        "hashtype": "md5",
        "hash": "4f83b78475e5582a014664915eb222d7",
        "size": 2471645608,
        "runtime_secret_name": "gcs-account",
    },
]


def sort_manifest(manifest):
    return sorted(manifest, key=lambda x: x["uid"])


def test_dolly():
    # fix the below models
    models = [
        truss_transfer.PyModelRepo(
            repo_id="gs://llama-3-2-1b-instruct/",
            revision="main",
            runtime_secret_name="gcs-account",
            volume_folder="julien_dummy",
            kind="gcs",
            ignore_patterns=["*.pth", "*cache*", "original*", "*.lock", "*.metadata"],
        )
    ]
    if not Path("/secrets/gcs-account").exists():
        pytest.skip(
            "Skipping test_dolly because GCS secret is not available. "
            "Please set up the GCS secret at /secrets/gcs-account."
        )

    print("Testing create_basetenpointer_from_models...")
    result = truss_transfer.create_basetenpointer_from_models(models)
    print("Success! Generated BasetenPointer manifest:")

    # Parse and pretty print the JSON
    manifest = json.loads(result)["pointers"]
    print(json.dumps(manifest, indent=2))
    # Test that the structure is correct
    assert len(manifest) == 10

    # Check the first pointer structure
    required_fields = [
        "resolution",
        "uid",
        "file_name",
        "hashtype",
        "hash",
        "size",
        "runtime_secret_name",
    ]
    resolution_fields = ["resolution_type", "path", "bucket_name"]

    for pointer, expected_pointer in zip(
        sort_manifest(manifest), sort_manifest(MANIFEST_EXPECTED)
    ):
        for field in required_fields:
            assert field in pointer, (
                f"Missing field '{field}' in pointer {pointer['uid']}"
            )
            assert pointer[field] == expected_pointer[field], (
                f"Field '{field}' mismatch in pointer {pointer['uid']}: {pointer[field]} != {expected_pointer[field]}"
            )
        for field in resolution_fields:
            assert (
                pointer["resolution"][field] == expected_pointer["resolution"][field]
            ), (
                f"Resolution field '{field}' mismatch in pointer {pointer['uid']}: "
                f"{pointer['resolution'][field]} != {expected_pointer['resolution'][field]}"
            )

    print("✓ BasetenPointer structure validation passed")
