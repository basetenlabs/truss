import json
import shutil
from pathlib import Path

import pytest
import truss_transfer

MANIFEST_EXPECTED = [
    {
        "resolution": {
            "resolution_type": "s3",
            "bucket_name": "bt-training-dev-org-b68c04fe47d34c85bfa91515bc9d5e2d",
            "key": "training_projects/n4q95w5/jobs/prwny3y/rank-0/checkpoint-24/tokenizer_config.json",
            "region": None,
        },
        "uid": "s3:bt-training-dev-org-b68c04fe47d34c85bfa91515bc9d5e2d:training_projects/n4q95w5/jobs/prwny3y/rank-0/checkpoint-24/tokenizer_config.json",
        "file_name": "/app/model_cache/julien_dummy/rank-0/checkpoint-24/tokenizer_config.json",
        "hashtype": "etag",
        "hash": "_1fece25e21776a0f1e96f725e18bdd7b_",
        "size": 1157008,
        "runtime_secret_name": "aws-secret-json",
    }
]


def sort_manifest(manifest):
    return sorted(manifest, key=lambda x: x["uid"])


def test_dolly():
    # fix the below models
    models = [
        truss_transfer.PyModelRepo(
            repo_id="s3://bt-training-dev-org-b68c04fe47d34c85bfa91515bc9d5e2d/training_projects/n4q95w5/jobs/prwny3y/",
            revision="",
            runtime_secret_name="aws-secret-json",
            volume_folder="julien_dummy",
            kind="s3",
            allow_patterns=["*tokenizer_config.json"],
            ignore_patterns=["*.pth", "*cache*", "original*", "*.lock", "*.metadata"],
        )
    ]
    if not Path("/secrets/aws-secret-json").exists():
        pytest.skip(
            "Skipping test_dolly because AWS secret is not available. "
            "Please set up the AWS secret at /secrets/aws-secret-json."
        )

    print("Testing create_basetenpointer_from_models...")
    result = truss_transfer.create_basetenpointer_from_models(models)
    print("Success! Generated BasetenPointer manifest:")
    # Parse and pretty print the JSON
    manifest = json.loads(result)["pointers"]
    print(json.dumps(manifest, indent=2))
    # Test that the structure is correct
    assert len(manifest) == 1

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
    resolution_fields = ["resolution_type", "key", "bucket_name", "region"]

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
    return result


def test_dolly_with_download():
    manifest = test_dolly()
    Path("/static-bptr").mkdir(parents=True, exist_ok=True)
    shutil.rmtree("/app/model_cache/julien_dummy", ignore_errors=True)
    with open("/static-bptr/static-bptr-manifest.json", "w") as f:
        f.write(manifest)

    print("✓ BasetenPointer manifest written to /static-bptr/static-bptr-manifest.json")
    truss_transfer.lazy_data_resolve("")
    print("✓ Data download via BasetenPointer successful")
    # check that tokenizer_config.json exists
    assert Path(
        "/app/model_cache/julien_dummy/rank-0/checkpoint-24/tokenizer_config.json"
    ).exists()
