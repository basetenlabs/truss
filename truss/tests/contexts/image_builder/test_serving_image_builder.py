import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from truss.contexts.image_builder.serving_image_builder import (
    HF_ACCESS_TOKEN_FILE_NAME,
    ServingImageBuilderContext,
    get_files_to_cache,
    update_model_key,
    update_model_name,
)
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_config import Build, ModelCache, ModelRepo, ModelServer, TrussConfig
from truss.truss_handle import TrussHandle

BASE_DIR = Path(__file__).parent


def test_serving_image_dockerfile_from_user_base_image(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    th.set_base_image("baseten/truss-server-base:3.9-v0.4.3", "/usr/local/bin/python3")
    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(th.spec.truss_dir)
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path)
        with open(tmp_path / "Dockerfile", "r") as f:
            gen_docker_lines = f.readlines()
        with open(
            f"{BASE_DIR}/../../../test_data/server.Dockerfile",
            "r",
        ) as f:
            server_docker_lines = f.readlines()

        def filter_empty_lines(lines):
            return list(filter(lambda x: x and x != "\n" and x != "", lines))

        gen_docker_lines = filter_empty_lines(gen_docker_lines)
        server_docker_lines = filter_empty_lines(server_docker_lines)
        assert gen_docker_lines == server_docker_lines


def test_requirements_setup_in_build_dir(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    th.add_python_requirement("numpy")
    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(th.spec.truss_dir)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path)
        with open(tmp_path / "requirements.txt", "r") as f:
            requirements_content = f.read()

        with open(f"{BASE_DIR}/../../../templates/server/requirements.txt", "r") as f:
            base_requirements_content = f.read()

        assert requirements_content == base_requirements_content + "numpy\n"


def test_overrides_model_id_vllm():
    config = TrussConfig(
        python_version="py39",
        build=Build(
            model_server=ModelServer.VLLM,
            arguments={"endpoint": "Completions", "model": "gs://llama-2-7b/"},
        ),
    )

    model_key = update_model_key(config)
    update_model_name(config, model_key)

    # Assert model overridden in config
    assert Path(config.build.arguments["model"]) == Path("/app/model_cache/llama-2-7b")
    assert config.model_cache == ModelCache(
        models=[ModelRepo(repo_id="gs://llama-2-7b/")]
    )


def test_overrides_model_id_tgi():
    config = TrussConfig(
        python_version="py39",
        build=Build(
            model_server=ModelServer.TGI,
            arguments={"endpoint": "Completions", "model_id": "gs://llama-2-7b/"},
        ),
    )

    model_key = update_model_key(config)
    update_model_name(config, model_key)

    # Assert model overridden in config
    assert Path(config.build.arguments["model_id"]) == Path(
        "/app/model_cache/llama-2-7b"
    )
    assert config.model_cache == ModelCache(
        models=[ModelRepo(repo_id="gs://llama-2-7b/")]
    )


def flatten_cached_files(local_cache_files):
    return [file.source for file in local_cache_files]


def test_correct_hf_files_accessed_for_caching():
    model = "openai/whisper-small"
    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(models=[ModelRepo(repo_id=model)]),
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        hf_path = Path("root/.cache/huggingface/hub")

        model_files, files_to_cache = get_files_to_cache(config, truss_path, build_path)
        files_to_cache = flatten_cached_files(files_to_cache)
        assert str(hf_path / "version.txt") in files_to_cache

        # It's unlikely the repo will change
        assert (
            str(
                hf_path
                / "models--openai--whisper-small/blobs/59ef8a839f271fa2183c6a4c302669d097e43b6d"
            )
            in files_to_cache
        )

        files = model_files[model]["files"]

        assert "model.safetensors" in files
        assert "tokenizer_config.json" in files


@patch("truss.contexts.image_builder.serving_image_builder.GCSCache.list_files")
def test_correct_gcs_files_accessed_for_caching(mock_list_bucket_files):
    mock_list_bucket_files.return_value = [
        "fake_model-001-of-002.bin",
        "fake_model-002-of-002.bin",
    ]
    model = "gs://crazy-good-new-model-7b"

    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(models=[ModelRepo(repo_id=model)]),
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_cache(config, truss_path, build_path)
        files_to_cache = flatten_cached_files(files_to_cache)

        assert (
            "/app/model_cache/crazy-good-new-model-7b/fake_model-001-of-002.bin"
            in files_to_cache
        )
        assert (
            "/app/model_cache/crazy-good-new-model-7b/fake_model-002-of-002.bin"
            in files_to_cache
        )

        assert "fake_model-001-of-002.bin" in model_files[model]["files"]
        assert "fake_model-001-of-002.bin" in model_files[model]["files"]


@patch("truss.contexts.image_builder.serving_image_builder.S3Cache.list_files")
def test_correct_s3_files_accessed_for_caching(mock_list_bucket_files):
    mock_list_bucket_files.return_value = [
        "fake_model-001-of-002.bin",
        "fake_model-002-of-002.bin",
    ]
    model = "s3://crazy-good-new-model-7b"

    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(models=[ModelRepo(repo_id=model)]),
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_cache(config, truss_path, build_path)
        files_to_cache = flatten_cached_files(files_to_cache)

        assert (
            "/app/model_cache/crazy-good-new-model-7b/fake_model-001-of-002.bin"
            in files_to_cache
        )
        assert (
            "/app/model_cache/crazy-good-new-model-7b/fake_model-002-of-002.bin"
            in files_to_cache
        )

        assert "fake_model-001-of-002.bin" in model_files[model]["files"]
        assert "fake_model-001-of-002.bin" in model_files[model]["files"]


@patch("truss.contexts.image_builder.serving_image_builder.GCSCache.list_files")
def test_correct_nested_gcs_files_accessed_for_caching(mock_list_bucket_files):
    mock_list_bucket_files.return_value = [
        "folder_a/folder_b/fake_model-001-of-002.bin",
        "folder_a/folder_b/fake_model-002-of-002.bin",
    ]
    model = "gs://crazy-good-new-model-7b/folder_a/folder_b"

    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(models=[ModelRepo(repo_id=model)]),
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_cache(config, truss_path, build_path)
        files_to_cache = flatten_cached_files(files_to_cache)

        assert (
            "/app/model_cache/crazy-good-new-model-7b/folder_a/folder_b/fake_model-001-of-002.bin"
            in files_to_cache
        )
        assert (
            "/app/model_cache/crazy-good-new-model-7b/folder_a/folder_b/fake_model-002-of-002.bin"
            in files_to_cache
        )

        assert (
            "folder_a/folder_b/fake_model-001-of-002.bin" in model_files[model]["files"]
        )
        assert (
            "folder_a/folder_b/fake_model-001-of-002.bin" in model_files[model]["files"]
        )


@patch("truss.contexts.image_builder.serving_image_builder.S3Cache.list_files")
def test_correct_nested_s3_files_accessed_for_caching(mock_list_bucket_files):
    mock_list_bucket_files.return_value = [
        "folder_a/folder_b/fake_model-001-of-002.bin",
        "folder_a/folder_b/fake_model-002-of-002.bin",
    ]
    model = "s3://crazy-good-new-model-7b/folder_a/folder_b"

    config = TrussConfig(
        python_version="py39",
        model_cache=ModelCache(models=[ModelRepo(repo_id=model)]),
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_cache(config, truss_path, build_path)
        files_to_cache = flatten_cached_files(files_to_cache)

        assert (
            "/app/model_cache/crazy-good-new-model-7b/folder_a/folder_b/fake_model-001-of-002.bin"
            in files_to_cache
        )
        assert (
            "/app/model_cache/crazy-good-new-model-7b/folder_a/folder_b/fake_model-002-of-002.bin"
            in files_to_cache
        )

        assert (
            "folder_a/folder_b/fake_model-001-of-002.bin" in model_files[model]["files"]
        )
        assert (
            "folder_a/folder_b/fake_model-001-of-002.bin" in model_files[model]["files"]
        )


@pytest.mark.integration
def test_tgi_caching_truss():
    with ensure_kill_all():
        truss_root = (
            Path(__file__).parent.parent.parent.parent.parent.resolve() / "truss"
        )
        truss_dir = truss_root / "test_data" / "test_tgi_truss"
        tr = TrussHandle(truss_dir)

        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        time.sleep(15)
        assert "Successfully downloaded weights." in container.logs()


@pytest.mark.integration
def test_vllm_caching_truss():
    with ensure_kill_all():
        truss_root = (
            Path(__file__).parent.parent.parent.parent.parent.resolve() / "truss"
        )
        truss_dir = truss_root / "test_data" / "test_vllm_truss"
        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)


@pytest.mark.integration
def test_truss_server_caching_truss():
    with ensure_kill_all():
        truss_root = (
            Path(__file__).parent.parent.parent.parent.parent.resolve() / "truss"
        )
        truss_dir = truss_root / "test_data" / "test_truss_server_caching_truss"
        tr = TrussHandle(truss_dir)

        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        time.sleep(15)
        assert "Downloading model.safetensors:" not in container.logs()


def test_model_cache_dockerfile():
    truss_root = Path(__file__).parent.parent.parent.parent.parent.resolve() / "truss"
    truss_dir = truss_root / "test_data" / "test_truss_server_caching_truss"
    tr = TrussHandle(truss_dir)

    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(tr.spec.truss_dir)

    secret_mount = f"RUN --mount=type=secret,id={HF_ACCESS_TOKEN_FILE_NAME}"
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path, use_hf_secret=True)
        with open(tmp_path / "Dockerfile", "r") as f:
            gen_docker_file = f.read()
            assert secret_mount in gen_docker_file
