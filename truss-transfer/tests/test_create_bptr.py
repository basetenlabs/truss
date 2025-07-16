import json

import truss_transfer

print(f"truss_transfer version: {truss_transfer.__version__}")


def test_dolly():
    # fix the below models
    models = [
        truss_transfer.ModelRepo(
            repo_id="julien-c/dummy-unknown",
            revision="60b8d3fe22aebb024b573f1cca224db3126d10f3",
            runtime_secret_name="hf_access_token_2",
            volume_folder="julien_dummy",
        )
    ]

    print("Testing create_basetenpointer_from_models...")
    result = truss_transfer.create_basetenpointer_from_models(models)
    print("Success! Generated BasetenPointer manifest:")

    # Parse and pretty print the JSON
    manifest = json.loads(result)
    print(json.dumps(manifest, indent=2))

    # Test that the structure is correct
    assert len(manifest) == 8

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
        assert models[0].revision in first_pointer["resolution"]["url"], (
            f"Revision mismatch in {field}"
        )

    print("✓ BasetenPointer structure validation passed")
