import json

import truss_transfer

print(f"truss_transfer version: {truss_transfer.__version__}")

HF_REPO = "julien-c/dummy-unknown"

HF_REVISION = "60b8d3fe22aebb024b573f1cca224db3126d10f3"

MANIFEST_EXPECTED = [
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/config.json",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:config.json",
        "file_name": "/app/model_cache/julien_dummy/config.json",
        "hashtype": "etag",
        "hash": "d9e9f15bc825e4b2c9249e9578f884bbcb5e3684",
        "size": 496,
        "runtime_secret_name": "hf_access_token_2",
    },
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/tf_model.h5",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:tf_model.h5",
        "file_name": "/app/model_cache/julien_dummy/tf_model.h5",
        "hashtype": "etag",
        "hash": "270a8b1acf605b95419149ae1c5a7a11ba13dc005d0ae582da2cdf4752251455",
        "size": 157312,
        "runtime_secret_name": "hf_access_token_2",
    },
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/merges.txt",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:merges.txt",
        "file_name": "/app/model_cache/julien_dummy/merges.txt",
        "hashtype": "etag",
        "hash": "d7c5738baaf4304ef4692f6fe8ad887b9517d047",
        "size": 36,
        "runtime_secret_name": "hf_access_token_2",
    },
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/.gitattributes",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:.gitattributes",
        "file_name": "/app/model_cache/julien_dummy/.gitattributes",
        "hashtype": "etag",
        "hash": "ae8c63daedbd4206d7d40126955d4e6ab1c80f8f",
        "size": 391,
        "runtime_secret_name": "hf_access_token_2",
    },
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/README.md",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:README.md",
        "file_name": "/app/model_cache/julien_dummy/README.md",
        "hashtype": "etag",
        "hash": "dd72d529da9961224818c8e9e7abbe8a57748d18",
        "size": 1114,
        "runtime_secret_name": "hf_access_token_2",
    },
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/vocab.json",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:vocab.json",
        "file_name": "/app/model_cache/julien_dummy/vocab.json",
        "hashtype": "etag",
        "hash": "9226d916ab5ccd75b5f9d921e317a70988c17a37",
        "size": 239,
        "runtime_secret_name": "hf_access_token_2",
    },
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/flax_model.msgpack",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:flax_model.msgpack",
        "file_name": "/app/model_cache/julien_dummy/flax_model.msgpack",
        "hashtype": "etag",
        "hash": "ff831ef9590de13abfb4caf6957b4dfff3930702c169f28a0fc47ff42b14aec5",
        "size": 58499,
        "runtime_secret_name": "hf_access_token_2",
    },
    {
        "resolution": {
            "url": "https://huggingface.co/julien-c/dummy-unknown/resolve/60b8d3fe22aebb024b573f1cca224db3126d10f3/pytorch_model.bin",
            "resolution_type": "http",
            "expiration_timestamp": 4044816725,
        },
        "uid": "julien-c/dummy-unknown:60b8d3fe22aebb024b573f1cca224db3126d10f3:pytorch_model.bin",
        "file_name": "/app/model_cache/julien_dummy/pytorch_model.bin",
        "hashtype": "etag",
        "hash": "0ef7dba7b1bda16ab1fcc8d67504190aa8d581854303d332ba3d0f1b42aefc5a",
        "size": 65074,
        "runtime_secret_name": "hf_access_token_2",
    },
]


def sort_manifest(manifest):
    return sorted(manifest, key=lambda x: x["uid"])


def test_dolly():
    # fix the below models
    models = [
        truss_transfer.PyModelRepo(
            repo_id=HF_REPO,
            revision=HF_REVISION,
            runtime_secret_name="hf_access_token_2",
            volume_folder="julien_dummy",
            kind="hf",
        )
    ]

    print("Testing create_basetenpointer_from_models...")
    result = truss_transfer.create_basetenpointer_from_models(models)
    print("Success! Generated BasetenPointer manifest:")

    # Parse and pretty print the JSON
    manifest = json.loads(result)["pointers"]
    print(json.dumps(manifest, indent=2))

    # Test that the structure is correct
    assert len(manifest) == 8

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
    resolution_fields = ["url", "resolution_type", "expiration_timestamp"]

    for pointer, expected in zip(
        sort_manifest(manifest), sort_manifest(MANIFEST_EXPECTED)
    ):
        for field in required_fields:
            assert field in pointer, f"Missing field: {field}"
            try:
                assert pointer[field] == expected[field], f"Field {field} mismatch"
            except AssertionError as e:
                print(f"Error in pointer {pointer['uid']}: {e}")

        assert models[0].revision in pointer["resolution"]["url"], (
            f"Revision mismatch in {field}"
        )
        for field in resolution_fields:
            assert field in pointer["resolution"], f"Missing resolution field: {field}"
            try:
                assert pointer["resolution"][field] == expected["resolution"][field], (
                    f"Resolution field {field} mismatch"
                )
            except AssertionError as e:
                print(f"Error in pointer {pointer['uid']}: {e}")

    print("✓ BasetenPointer structure validation passed")


def test_qwen3():
    models = [
        truss_transfer.PyModelRepo(
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            revision="main",
            runtime_secret_name="hf_access_token_2",
            volume_folder="julien_dummy",
            kind="hf",
        )
    ]

    print("Testing create_basetenpointer_from_models...")
    result = truss_transfer.create_basetenpointer_from_models(models)
    print("Success! Generated BasetenPointer manifest:")

    # Parse and pretty print the JSON
    manifest = json.loads(result)

    print(json.dumps(manifest, indent=2))
