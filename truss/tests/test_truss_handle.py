import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from python_on_whales.exceptions import DockerException
from tenacity import RetryError

from truss.base.custom_types import Example
from truss.base.errors import ContainerIsDownError, ContainerNotFoundError
from truss.base.truss_config import map_local_to_supported_python_version
from truss.local.local_config_handler import LocalConfigHandler
from truss.templates.control.control.helpers.custom_types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
)
from truss.tests.test_testing_utilities_for_other_tests import (
    ensure_kill_all,
    kill_all_with_retries,
)
from truss.truss_handle.patch.custom_types import PatchRequest
from truss.truss_handle.truss_handle import DockerURLs, TrussHandle, wait_for_truss
from truss.util.docker import Docker, DockerStates


def test_spec(custom_model_truss_dir_with_pre_and_post):
    dir_path = custom_model_truss_dir_with_pre_and_post
    th = TrussHandle(dir_path)
    spec = th.spec
    assert spec.truss_dir == dir_path


def test_description(custom_model_truss_dir_with_pre_and_post_description):
    dir_path = custom_model_truss_dir_with_pre_and_post_description
    th = TrussHandle(dir_path)
    spec = th.spec
    assert spec.description == "This model adds 3 to all inputs"


def test_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    resp = th.predict([1, 2, 3, 4])
    assert resp == {"predictions": [4, 5, 6, 7]}


def test_predict_with_external_packages(custom_model_with_external_package):
    th = TrussHandle(custom_model_with_external_package)
    resp = th.predict([1, 2, 3, 4])
    assert resp == [1, 1, 1, 1]


def test_server_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    resp = th.server_predict([1, 2, 3, 4])
    assert resp == {"predictions": [4, 5, 6, 7]}


def test_readme_generation_int_example(
    test_data_path, custom_model_truss_dir_with_pre_and_post
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace("\n", "")
    correct_readme_contents = _read_readme(test_data_path / "readme_int_example.md")
    assert readme_contents == correct_readme_contents


def test_readme_generation_no_example(
    test_data_path, custom_model_truss_dir_with_pre_and_post_no_example
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post_no_example)
    if os.path.exists(th._spec.examples_path):
        # Remove the examples file
        os.remove(th._spec.examples_path)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace("\n", "")
    correct_readme_contents = _read_readme(test_data_path / "readme_no_example.md")
    assert readme_contents == correct_readme_contents


def test_readme_generation_str_example(
    test_data_path, custom_model_truss_dir_with_pre_and_post_str_example
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post_str_example)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace("\n", "")
    correct_readme_contents = _read_readme(test_data_path / "readme_str_example.md")
    assert readme_contents == correct_readme_contents


@pytest.mark.integration
def test_build_docker_image(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-build-image-tag:0.0.1"
    image = th.build_serving_docker_image(tag=tag)
    assert image.repo_tags[0] == tag


@pytest.mark.integration
@pytest.mark.parametrize(
    "base_image, path, expected_fail",
    [
        ("baseten/truss-server-base:3.9-v0.4.8rc4", "/usr/local/bin/python3", False),
        ("python:3.8", "/usr/local/bin/python3", False),
        ("python:3.10", "/usr/local/bin/python3", False),
        ("python:3.11", "/usr/local/bin/python3", False),
        ("python:3.13", "/usr/local/bin/python3", False),
        ("python:alpine", "/usr/local/bin/python3", True),
        ("python:2.7-slim", "/usr/local/bin/python", True),
        ("python:3.7-slim", "/usr/local/bin/python3", True),
    ],
)
def test_build_serving_docker_image_from_user_base_image_live_reload(
    custom_model_truss_dir, base_image, path, expected_fail
):
    th = TrussHandle(custom_model_truss_dir)
    th.set_base_image(base_image, path)
    th.live_reload()
    try:
        th.build_serving_docker_image(cache=False)
    except DockerException as exc:
        assert expected_fail is True
        assert "It returned with code 1" in str(exc)


@pytest.mark.integration
def test_docker_predict_custom_base_image(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.set_base_image(
        "wallies/python-cuda:3.10-cuda11.7-runtime", "/usr/local/bin/python"
    )
    with ensure_kill_all():
        result = th.docker_predict([1, 2], local_port=None)
        assert result == {"predictions": [4, 5]}


@pytest.mark.integration
def test_build_docker_image_gpu(custom_model_truss_dir_for_gpu, tmp_path):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    tag = "test-build-image-gpu-tag:0.0.1"
    build_dir = tmp_path / "scaffold_build_dir"
    image = th.build_serving_docker_image(tag=tag, build_dir=build_dir)
    assert image.repo_tags[0] == tag


@pytest.mark.integration
def test_build_docker_image_control_gpu(custom_model_truss_dir_for_gpu, tmp_path):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    th.live_reload(True)
    tag = "test-build-image-control-gpu-tag:0.0.1"
    build_dir = tmp_path / "scaffold_build_dir"
    image = th.build_serving_docker_image(tag=tag, build_dir=build_dir)
    assert image.repo_tags[0] == tag


@pytest.mark.integration
def test_docker_run(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-docker-run-tag:0.0.1"
    container = th.docker_run(tag=tag, local_port=None)
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.skip(reason="Needs gpu")
@pytest.mark.integration
def test_docker_run_gpu(custom_model_truss_dir_for_gpu):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    tag = "test-docker-run-gpu-tag:0.0.1"
    container = th.docker_run(tag=tag, local_port=None)
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def test_docker_run_without_tag(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    container = th.docker_run(local_port=None)
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def get_docker_containers_from_labels(custom_model_truss_dir_with_pre_and_post):
    with ensure_kill_all():
        t1 = TrussHandle(custom_model_truss_dir_with_pre_and_post)
        assert len(t1.get_serving_docker_containers_from_labels()) == 0
        t1.docker_run(local_port=None)
        assert len(t1.get_serving_docker_containers_from_labels()) == 1
        t1.docker_run(local_port=None)
        assert len(t1.get_serving_docker_containers_from_labels()) == 2
        t1.kill_container()
        assert len(t1.get_serving_docker_containers_from_labels()) == 0


@pytest.mark.integration
def test_predict_use_docker(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-docker-predict-tag:0.0.1"
    with ensure_kill_all():
        result = th.predict([1, 2], tag=tag, use_docker=True)
        assert result == {"predictions": [4, 5]}


@pytest.mark.integration
def test_docker_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-docker-predict-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1, 2], tag=tag, local_port=None)
        assert result == {"predictions": [4, 5]}


@pytest.mark.integration
def test_docker_predict_model_with_external_packages(
    custom_model_with_external_package,
):
    th = TrussHandle(custom_model_with_external_package)
    tag = "test-docker-predict-ext-pkg-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1, 2], tag=tag, local_port=None)
        assert result == [1, 1]


@pytest.mark.integration
def test_docker_predict_with_bundled_packages(
    custom_model_truss_dir_with_bundled_packages,
):
    th = TrussHandle(custom_model_truss_dir_with_bundled_packages)
    tag = "test-docker-predict-bundled-packages-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1, 2], tag=tag, local_port=None)
        assert result == {"predictions": [1]}


@pytest.mark.integration
def test_docker_multiple_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-docker-predict-tag:0.0.1"
    with ensure_kill_all():
        r1 = th.docker_predict([1, 2], tag=tag, local_port=None)
        r2 = th.docker_predict([3, 4], tag=tag, local_port=None)
        assert r1 == {"predictions": [4, 5]}
        assert r2 == {"predictions": [6, 7]}
        assert len(th.get_serving_docker_containers_from_labels()) == 1


@pytest.mark.integration
def test_kill_all(custom_model_truss_dir, custom_model_truss_dir_with_pre_and_post):
    t1 = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    t2 = TrussHandle(custom_model_truss_dir)
    with ensure_kill_all():
        t1.docker_run(local_port=None)
        assert len(t1.get_serving_docker_containers_from_labels()) == 1
        t2.docker_run(local_port=None)
        assert len(t2.get_serving_docker_containers_from_labels()) == 1
        kill_all_with_retries()
        assert len(t1.get_serving_docker_containers_from_labels()) == 0
        assert len(t2.get_serving_docker_containers_from_labels()) == 0


@pytest.mark.skip(reason="Needs gpu")
@pytest.mark.integration
def test_docker_predict_gpu(custom_model_truss_dir_for_gpu):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    tag = "test-docker-predict-gpu-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result["predictions"][0]["cuda_version"].startswith("11")


@pytest.mark.integration
def test_docker_predict_secrets(custom_model_truss_dir_for_secrets):
    th = TrussHandle(custom_model_truss_dir_for_secrets)
    tag = "test-docker-predict-secrets-tag:0.0.1"
    LocalConfigHandler.set_secret("secret_name", "secret_value")
    with ensure_kill_all():
        try:
            result = th.docker_predict(
                {"instances": ["secret_name"]}, tag=tag, local_port=None
            )
            assert result["predictions"][0] == "secret_value"
        finally:
            LocalConfigHandler.remove_secret("secret_name")


@pytest.mark.integration
def test_docker_no_preprocess_custom_model(no_preprocess_custom_model):
    th = TrussHandle(no_preprocess_custom_model)
    tag = "test-docker-no-preprocess-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result["predictions"][0] == 2


@pytest.mark.integration
def test_docker_long_load(long_load_model):
    th = TrussHandle(long_load_model)
    tag = "test-docker-long-load-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result["predictions"][0] == 1


@pytest.mark.integration
def test_local_no_preprocess_custom_model(no_preprocess_custom_model):
    th = TrussHandle(no_preprocess_custom_model)
    result = th.server_predict([1])
    assert result["predictions"][0] == 2


@pytest.mark.integration
def test_docker_no_postprocess_custom_model(no_postprocess_custom_model):
    th = TrussHandle(no_postprocess_custom_model)
    tag = "test-docker-no-postprocess-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result["predictions"][0] == 2


@pytest.mark.integration
def test_local_no_postprocess_custom_model(no_postprocess_custom_model):
    th = TrussHandle(no_postprocess_custom_model)
    result = th.server_predict([1])
    assert result["predictions"][0] == 2


@pytest.mark.integration
def test_docker_no_load_custom_model(no_load_custom_model):
    th = TrussHandle(no_load_custom_model)
    tag = "test-docker-no-load-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result["predictions"][0] == 1


@pytest.mark.integration
def test_local_no_load_custom_model(no_load_custom_model):
    th = TrussHandle(no_load_custom_model)
    result = th.server_predict([1])
    assert result["predictions"][0] == 1


@pytest.mark.integration
def test_docker_no_params_init_custom_model(no_params_init_custom_model):
    th = TrussHandle(no_params_init_custom_model)
    tag = "test-docker-no-params-init-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result["predictions"][0] == 1


@pytest.mark.integration
def test_local_no_params_init_custom_model(no_params_init_custom_model):
    th = TrussHandle(no_params_init_custom_model)
    result = th.server_predict([1])
    assert result["predictions"][0] == 1


@pytest.mark.integration
def test_custom_python_requirement(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_python_requirement("theano")
    th.add_python_requirement("scipy")
    tag = "test-custom-python-req-tag:0.0.1"
    container = th.docker_run(tag=tag, local_port=None)
    try:
        verify_python_requirement_installed_on_container(container, "theano")
        verify_python_requirement_installed_on_container(container, "scipy")
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def test_custom_system_package(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_system_package("jq")
    th.add_system_package("fzf")
    tag = "test-custom-system-package-tag:0.0.1"
    container = th.docker_run(tag=tag, local_port=None)
    try:
        verify_system_package_installed_on_container(container, "jq")
        verify_system_package_installed_on_container(container, "fzf")
    finally:
        Docker.client().kill(container)


@pytest.mark.parametrize(
    "python_version, expected_python_version",
    [("3.8", "py38"), ("py38", "py38"), ("3.9", "py39"), ("py39", "py39")],
)
def test_update_python_version(
    python_version, expected_python_version, custom_model_truss_dir_with_pre_and_post
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.update_python_version(python_version)
    assert th.spec.python_version == expected_python_version


def test_update_requirements(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    requirements = ["tensorflow==2.3.1", "uvicorn==0.12.2"]
    th.update_requirements(requirements)
    sc_requirements = th.spec.requirements
    assert sc_requirements == requirements


def test_update_requirements_from_file(
    custom_model_truss_dir_with_pre_and_post, tmp_path
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    file_requirements = [
        "tensorflow==2.3.1",
        "# this is comment. Please don't add.",
        "    # this is comment with a big space. Please don't add.",
        "uvicorn==0.12.2",
    ]
    allowed_requirements = ["tensorflow==2.3.1", "uvicorn==0.12.2"]
    req_file_path = tmp_path / "requirements.txt"
    with req_file_path.open("w") as req_file:
        for req in file_requirements:
            req_file.write(f"{req}\n")
    th.update_requirements_from_file(str(req_file_path))
    sc_requirements = th.spec.requirements
    assert sc_requirements == allowed_requirements


@pytest.mark.integration
def test_add_environment_variable(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_environment_variable("test_env", "test_value")
    tag = "test-add-env-var-tag:0.0.1"
    container = th.docker_run(tag=tag, local_port=None)
    try:
        verify_environment_variable_on_container(container, "test_env", "test_value")
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def test_build_commands(test_data_path):
    truss_dir = test_data_path / "test_build_commands"
    tr = TrussHandle(truss_dir)
    with ensure_kill_all():
        r1 = tr.docker_predict([1, 2], local_port=None)
        assert r1 == {"predictions": [1, 2]}


@pytest.mark.integration
def test_build_commands_failure(test_data_path):
    truss_dir = test_data_path / "test_build_commands_failure"
    tr = TrussHandle(truss_dir)
    try:
        tr.docker_run(local_port=None, detach=True, wait_for_server_ready=True)
    except DockerException as exc:
        assert "It returned with code 1" in str(exc)


def test_add_data_file(custom_model_truss_dir_with_pre_and_post, tmp_path):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    data_filepath = tmp_path / "test_data.txt"
    with data_filepath.open("w") as data_file:
        data_file.write("test")

    th.add_data(str(data_filepath))

    scaf_data_filepath = th.spec.data_dir / "test_data.txt"
    assert scaf_data_filepath.exists()
    with scaf_data_filepath.open() as data_file:
        assert data_file.read() == "test"


def test_add_data_fileglob(custom_model_truss_dir_with_pre_and_post, tmp_path):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    file1_path = tmp_path / "test_data1.txt"
    with file1_path.open("w") as data_file:
        data_file.write("test1")

    file2_path = tmp_path / "test_data2.txt"
    with file2_path.open("w") as data_file:
        data_file.write("test2")

    file2_path = tmp_path / "test_data3.json"
    with file2_path.open("w") as data_file:
        data_file.write("{}")

    th.add_data(f"{str(tmp_path)}/*.txt")

    assert (th.spec.data_dir / "test_data1.txt").exists()
    assert (th.spec.data_dir / "test_data2.txt").exists()
    assert not (th.spec.data_dir / "test_data2.json").exists()


def test_add_data_dir(custom_model_truss_dir_with_pre_and_post, tmp_path):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    sub_dir = tmp_path / "sub"
    sub_sub_dir = sub_dir / "sub"
    sub_sub_dir.mkdir(parents=True)

    file_path = sub_sub_dir / "test_file.txt"
    with file_path.open("w") as data_file:
        data_file.write("test")

    th.add_data(str(sub_dir))

    scaf_file_path = th.spec.data_dir / "sub" / "sub" / "test_file.txt"
    assert scaf_file_path.exists()
    with scaf_file_path.open() as data_file:
        assert data_file.read() == "test"


def test_examples(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    examples = th.examples()
    assert "example1" in [example.name for example in examples]


def test_add_example_new(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    orig_examples = th.examples()
    th.add_example("example2", [[1]])
    assert th.examples() == [*orig_examples, Example("example2", [[1]])]


def test_add_example_update(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_example("example1", [[1]])
    assert th.examples() == [Example("example1", [[1]])]


def test_model_without_pre_post(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    resp = th.server_predict([1, 2, 3, 4])
    assert resp == [1, 1, 1, 1]


@pytest.mark.integration
def test_docker_predict_model_without_pre_post(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    with ensure_kill_all():
        resp = th.docker_predict([1, 2, 3, 4], local_port=None)
        assert resp == [1, 1, 1, 1]


@pytest.mark.integration
def test_control_truss_apply_patch(custom_model_control):
    th = TrussHandle(custom_model_control)
    tag = "test-docker-custom-model-control-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result[0] == 1

        running_hash = th.truss_hash_on_serving_container()
        new_model_code = """
class Model:
    def predict(self, model_input):
        return [2 for i in model_input]
"""
        patch_request = PatchRequest(
            hash="dummy",
            prev_hash=running_hash,
            patches=[
                Patch(
                    type=PatchType.MODEL_CODE,
                    body=ModelCodePatch(
                        action=Action.UPDATE, path="model.py", content=new_model_code
                    ),
                )
            ],
        )

        th.patch_container(patch_request)
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result[0] == 2


@pytest.mark.integration
def test_regular_truss_local_update_flow(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    tag = "test-docker-custom-model-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result[0] == 1
        orig_num_truss_images = len(th.get_all_docker_images())

        # No new docker images on second predict
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert orig_num_truss_images == len(th.get_all_docker_images())

        with (custom_model_truss_dir / "model" / "model.py").open(
            "w"
        ) as model_code_file:
            model_code_file.write(
                """
class Model:
    def predict(self, model_input):
        return [2 for i in model_input]
"""
            )
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result[0] == 2
        # A new image should have been created
        assert len(th.get_all_docker_images()) == orig_num_truss_images + 1


@patch("truss.truss_handle.truss_handle.directory_content_hash")
def test_truss_hash_caching_based_on_max_mod_time(
    directory_content_patcher, custom_model_truss_dir
):
    directory_content_patcher.return_value = "mock_hash"
    th = TrussHandle(custom_model_truss_dir)
    labels = th._get_serving_labels()
    labels2 = th._get_serving_labels()
    assert labels == labels2
    directory_content_patcher.assert_called_once()

    time.sleep(0.1)  # Make sure different mod time
    (custom_model_truss_dir / "model" / "model.py").touch()
    labels3 = th._get_serving_labels()
    assert labels3 != labels
    directory_content_patcher.call_count == 2


@patch("truss.truss_handle.truss_handle.get_docker_urls")
@patch("truss.truss_handle.truss_handle.get_container_state")
def test_container_oom_caught_during_waiting(
    container_state_mock, get_docker_urls_mock
):
    container_state_mock.return_value = DockerStates.OOMKILLED
    get_docker_urls_mock.return_value = DockerURLs("http://localhost:8080")
    with pytest.raises(ContainerIsDownError):
        wait_for_truss(container=MagicMock())


@patch("truss.truss_handle.truss_handle.get_docker_urls")
@patch("truss.truss_handle.truss_handle.get_container_state")
@pytest.mark.integration
def test_container_stuck_in_created(container_state_mock, get_docker_urls_mock):
    container_state_mock.return_value = DockerStates.CREATED
    get_docker_urls_mock.return_value = DockerURLs("http://localhost:8080")
    with pytest.raises(ContainerIsDownError):
        wait_for_truss(container=MagicMock())


@pytest.mark.integration
def test_control_truss_local_update_that_crashes_inference_server(custom_model_control):
    th = TrussHandle(custom_model_control)
    tag = "test-docker-custom-model-control-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result[0] == 1

        bad_model_code = """
class Model:
    def malformed
"""
        model_code_file_path = custom_model_control / "model" / "model.py"
        with model_code_file_path.open("w") as model_code_file:
            model_code_file.write(bad_model_code)
        with pytest.raises(RetryError) as exc_info:
            th.docker_predict([1], tag=tag, local_port=None)
        resp = exc_info.value.last_attempt.result()
        assert resp.status_code == 503
        assert (
            "Model load failed" in resp.text
            or "It appears your model has stopped running" in resp.text
        )

        # Should be able to fix code after
        good_model_code = """
class Model:
    def predict(self, model_input):
        return [2 for i in model_input]
"""
        with model_code_file_path.open("w") as model_code_file:
            model_code_file.write(good_model_code)
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result[0] == 2


@pytest.mark.integration
@pytest.mark.parametrize(
    "patch_path, expected_call_count",
    [("hash_is_current", 1), ("hash_is_current_but_only_every_third_call_succeeds", 3)],
)
def test_patch_ping_flow(
    patch_path, expected_call_count, custom_model_control, patch_ping_test_server
):
    port = patch_ping_test_server
    patch_ping_url = f"http://host.docker.internal:{port}/{patch_path}"
    th = TrussHandle(custom_model_control)
    tag = "test-docker-custom-model-control-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict(
            [1], tag=tag, patch_ping_url=patch_ping_url, local_port=None
        )
        assert result == [1]

        # Make sure the patch ping url was actually hit
        stats = requests.get(f"http://127.0.0.1:{port}/stats").json()
        assert stats[f"{patch_path}_called_count"] == expected_call_count


def test_handle_if_container_dne(custom_model_truss_dir):
    def return_container_dne(self):
        return "DNE"

    with (
        patch.object(TrussHandle, "_try_patch", new=return_container_dne),
        pytest.raises(ContainerNotFoundError),
    ):
        truss_handle = TrussHandle(truss_dir=custom_model_truss_dir)
        truss_handle.docker_run(local_port=None)
    kill_all_with_retries()


def test_docker_predict_container_does_not_exist(custom_model_truss_dir):
    def return_container_dne(self):
        return "DNE"

    with (
        patch.object(TrussHandle, "_try_patch", new=return_container_dne),
        pytest.raises(ContainerNotFoundError),
    ):
        truss_handle = TrussHandle(truss_dir=custom_model_truss_dir)
        truss_handle.docker_predict([1], local_port=None)
    kill_all_with_retries()


@pytest.mark.integration
def test_external_data(custom_model_external_data_access_tuple_fixture):
    truss_dir, expected_content = custom_model_external_data_access_tuple_fixture
    th = TrussHandle(truss_dir)
    tag = "test-external-data-access-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([], tag=tag, network="host", local_port=None)
        assert result == expected_content


@pytest.mark.integration
def test_external_data_gpu(custom_model_external_data_access_tuple_fixture_gpu):
    truss_dir, expected_content = custom_model_external_data_access_tuple_fixture_gpu
    th = TrussHandle(truss_dir)
    tag = "test-external-data-access-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict([], tag=tag, network="host", local_port=None)
        assert result == expected_content


def _container_exists(container) -> bool:
    for row in Docker.client().ps():
        if row.id.startswith(container.id):
            return True
    return False


def verify_system_package_installed_on_container(container, pkg: str):
    resp = container.execute(["which", pkg])
    assert resp.strip() == f"/usr/bin/{pkg}"


def verify_system_requirement_not_installed_on_container(container, pkg: str):
    try:
        container.execute(["dpkg", "-l", pkg])
    except DockerException as excp:
        assert "no packages found" in str(excp)


def verify_python_requirement_installed_on_container(container, req: str):
    resp = container.execute(["pip", "show", req])
    assert resp.splitlines()[0].lower() == f"Name: {req}".lower()


def verify_python_requirement_not_installed_on_container(container, req: str):
    try:
        container.execute(["pip", "show", req])
    except DockerException as excp:
        assert "not found" in str(excp)


def verify_environment_variable_on_container(
    container, env_var_name: str, env_var_value: str
):
    resp = container.execute(["env"])
    needle = f"{env_var_name}={env_var_value}"
    assert needle in resp.splitlines()


def _read_readme(readme_correct_path: Path) -> str:
    return readme_correct_path.open().read().replace("\n", "")


def generate_default_config():
    # The test fixture varies with host version.
    python_version = map_local_to_supported_python_version()
    config = {
        "python_version": python_version,
        "resources": {
            "accelerator": None,
            "cpu": "1",
            "memory": "2Gi",
            "use_gpu": False,
        },
    }
    return config


def test_config_verbose(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

    new_config = generate_default_config()
    assert new_config == th.spec.config.to_dict(verbose=False)

    th.live_reload()
    new_config["live_reload"] = True
    assert new_config == th.spec.config.to_dict(verbose=False)
