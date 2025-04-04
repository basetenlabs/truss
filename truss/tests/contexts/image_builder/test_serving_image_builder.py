import filecmp
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from truss.base.constants import (
    BASE_TRTLLM_REQUIREMENTS,
    TRTLLM_BASE_IMAGE,
    TRTLLM_PREDICT_CONCURRENCY,
    TRTLLM_PYTHON_EXECUTABLE,
    TRTLLM_TRUSS_DIR,
)
from truss.base.truss_config import ModelCache, ModelRepo, TrussConfig
from truss.contexts.image_builder.serving_image_builder import (
    HF_ACCESS_TOKEN_FILE_NAME,
    ServingImageBuilderContext,
    get_files_to_model_cache_v1,
)
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle.truss_handle import TrussHandle

BASE_DIR = Path(__file__).parent


@patch("platform.machine", return_value="amd")
def test_serving_image_dockerfile_from_user_base_image(
    mock_machine, test_data_path, custom_model_truss_dir
):
    th = TrussHandle(custom_model_truss_dir)
    # The test fixture python varies with host version, need to pin here.
    th.update_python_version("py39")
    th.set_base_image("baseten/truss-server-base:3.9-v0.4.3", "/usr/local/bin/python3")
    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(th.spec.truss_dir)
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path)
        with open(tmp_path / "Dockerfile", "r") as f:
            gen_docker_lines = f.readlines()
        with open(test_data_path / "server.Dockerfile", "r") as f:
            server_docker_lines = f.readlines()

        # Remove both empty lines + comments
        def filter_unneeded_lines(lines):
            return [x for x in lines if x.strip() and not x.strip().startswith("#")]

        gen_docker_lines = filter_unneeded_lines(gen_docker_lines)
        server_docker_lines = filter_unneeded_lines(server_docker_lines)
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


def flatten_cached_files(local_cache_files):
    return [file.source for file in local_cache_files]


def test_correct_hf_files_accessed_for_caching():
    model = "openai/whisper-small"
    config = TrussConfig(
        python_version="py39", model_cache=ModelCache([ModelRepo(repo_id=model)])
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        hf_path = Path("root/.cache/huggingface/hub")

        model_files, files_to_cache = get_files_to_model_cache_v1(
            config, truss_path, build_path
        )
        files_to_cache = flatten_cached_files(files_to_cache)
        assert str(hf_path / "version.txt") in files_to_cache

        blobs = [
            blob
            for blob in files_to_cache
            if blob.startswith(f"{hf_path}/models--openai--whisper-small/blobs/")
        ]
        assert len(blobs) >= 1

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
        python_version="py39", model_cache=ModelCache([ModelRepo(repo_id=model)])
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_model_cache_v1(
            config, truss_path, build_path
        )
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
        python_version="py39", model_cache=ModelCache([ModelRepo(repo_id=model)])
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_model_cache_v1(
            config, truss_path, build_path
        )
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
        python_version="py39", model_cache=ModelCache([ModelRepo(repo_id=model)])
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_model_cache_v1(
            config, truss_path, build_path
        )
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
        python_version="py39", model_cache=ModelCache([ModelRepo(repo_id=model)])
    )

    with TemporaryDirectory() as tmp_dir:
        truss_path = Path(tmp_dir)
        build_path = truss_path / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        model_files, files_to_cache = get_files_to_model_cache_v1(
            config, truss_path, build_path
        )
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
def test_test_truss_server_model_cache_v1(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_truss_server_model_cache_v1"
        tr = TrussHandle(truss_dir)

        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True, network="host"
        )
        time.sleep(15)
        assert "Downloading model.safetensors:" not in container.logs()


@pytest.mark.integration
def test_test_truss_server_model_cache_v2(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_truss_server_model_cache_v2"
        tr = TrussHandle(truss_dir)

        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True, network="host"
        )
        time.sleep(15)
        assert container.logs()


def test_model_cache_dockerfile(test_data_path):
    truss_dir = test_data_path / "test_truss_server_model_cache_v1"
    tr = TrussHandle(truss_dir)
    assert tr.spec.config.model_cache.is_v1
    assert not tr.spec.config.model_cache.is_v2
    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(tr.spec.truss_dir)

    secret_mount = f"RUN --mount=type=secret,id={HF_ACCESS_TOKEN_FILE_NAME}"
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path, use_hf_secret=True)
        with open(tmp_path / "Dockerfile", "r") as f:
            gen_docker_file = f.read()
            assert secret_mount in gen_docker_file
            assert "cache_warmer.py" in gen_docker_file
            assert "bptr-manifest" not in gen_docker_file


def test_model_cache_dockerfile_v2(test_data_path):
    truss_dir = test_data_path / "test_truss_server_model_cache_v2"
    tr = TrussHandle(truss_dir)
    assert not tr.spec.config.model_cache.is_v1
    assert tr.spec.config.model_cache.is_v2
    # assert tr.spec.config.model_cache.is_v2
    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(tr.spec.truss_dir)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path, use_hf_secret=True)
        assert (tmp_path / "bptr-manifest").exists(), "bptr-manifest not found"
        with open(tmp_path / "bptr-manifest", "r") as f:
            json_bptr = json.load(f)["pointers"]
        print(json_bptr)
        assert len(json_bptr) == 7, (
            f"bptr-manifest should have 7 entries, found {len(json_bptr)}"
        )
        assert (
            json_bptr[0]["file_name"]
            == "/app/model_cache/julien_c_esper/.gitattributes"
        )
        assert json_bptr[0]["hash"] == "602b71f15d40ed68c5f96330e3f3175a76a32126"
        assert json_bptr[0]["size"] == 445
        assert (
            json_bptr[0]["resolution"]["url"]
            == "https://huggingface.co/julien-c/EsperBERTo-small/resolve/4c7798256a4a6d577738150840c8f728361496d6/.gitattributes"
        )
        assert (
            json_bptr[0]["resolution"]["expiration_timestamp"]
            > time.time() + 20 * 365 * 24 * 60 * 60
        ), (
            f"Expected unix expiration timestamp to be at least 20 years ahead, but got {json_bptr[0]['resolution']['expiration_timestamp']}. "
        )

        # lfs files
        assert (
            json_bptr[4]["file_name"]
            == "/app/model_cache/julien_c_esper/model.safetensors"
        )
        assert (
            json_bptr[4]["hash"]
            == "78ee94168f400dd136a1418a9f21f01ada049cdb3c064145b1400642cf342de6"
        )
        assert json_bptr[4]["size"] == 336392830
        assert (
            json_bptr[4]["resolution"]["url"]
            == "https://huggingface.co/julien-c/EsperBERTo-small/resolve/4c7798256a4a6d577738150840c8f728361496d6/model.safetensors"
        )

        with open(tmp_path / "Dockerfile", "r") as f:
            gen_docker_file = f.read()
            print(gen_docker_file)
            assert "truss-transfer" in gen_docker_file
            assert "COPY ./bptr-manifest /bptr/bptr-manifest" in gen_docker_file, (
                "bptr-manifest copy not found in Dockerfile"
            )
            assert "cache_warmer.py" not in gen_docker_file


def test_ignore_files_during_build_setup(custom_model_truss_dir_with_truss_ignore):
    th = TrussHandle(custom_model_truss_dir_with_truss_ignore)

    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(th.spec.truss_dir)

    ignore_files = ["random_file_1.txt"]
    ignore_folder = "random_folder_1/"
    do_not_ignore_folder = "random_folder_2/"

    with TemporaryDirectory() as build_dir:
        build_path = Path(build_dir)
        image_builder.prepare_image_build_dir(build_path)

        for file in ignore_files:
            assert not (build_path / file).exists()

        assert not (build_path / ignore_folder).exists()
        assert (build_path / do_not_ignore_folder).exists()


def test_trt_llm_build_dir(custom_model_trt_llm):
    th = TrussHandle(custom_model_trt_llm)
    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(th.spec.truss_dir)
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path)
        build_th = TrussHandle(tmp_path)

        # Check that all files were copied
        _assert_copied(
            TRTLLM_TRUSS_DIR / "src", tmp_path / "server" / "extensions" / "trt_llm"
        )
        _assert_copied(TRTLLM_TRUSS_DIR / "packages", tmp_path / "packages")

        assert (
            build_th.spec.config.runtime.predict_concurrency
            == TRTLLM_PREDICT_CONCURRENCY
        )
        assert build_th.spec.config.base_image.image == TRTLLM_BASE_IMAGE
        assert (
            build_th.spec.config.base_image.python_executable_path
            == TRTLLM_PYTHON_EXECUTABLE
        )
        assert BASE_TRTLLM_REQUIREMENTS == build_th.spec.config.requirements


def _assert_copied(src_path: str, dest_path: str):
    for dirpath, dirnames, filenames in os.walk(src_path):
        rel_path = os.path.relpath(dirpath, src_path)
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dest_file = os.path.join(dest_path, rel_path, filename)
            assert os.path.exists(dest_file), f"{dest_file} was not copied"
            assert filecmp.cmp(src_file, dest_file, shallow=False), (
                f"{src_file} and {dest_file} are not the same"
            )
