import json
from pathlib import Path

import pytest
import truss_transfer


def test_dolly():
    # fix the below models
    models = [
        truss_transfer.PyModelRepo(
            repo_id="gs://llama-3-2-1b-instruct/",
            revision="main",
            runtime_secret_name="gcs-account",
            volume_folder="julien_dummy",
            kind="gcs",
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
    manifest = json.loads(result)
    print(json.dumps(manifest, indent=2))

    # Test that the structure is correct
    assert len(manifest) == 40

    # Check the first pointer structure
    first_pointer = manifest[0]
    required_fields = [
        "resolution",
        "uid",
        "file_name",
        "hashtype",
        "hash",
        "size",
        "runtime_secret_name",
    ]

    for field in required_fields:
        assert field in first_pointer, f"Missing field: {field}"

    print("✓ BasetenPointer structure validation passed")
