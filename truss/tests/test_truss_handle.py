import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from python_on_whales.exceptions import DockerException
from truss.docker import Docker, DockerStates
from truss.errors import ContainerIsDownError, ContainerNotFoundError
from truss.local.local_config_handler import LocalConfigHandler
from truss.templates.control.control.helpers.types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
)
from truss.tests.test_testing_utilities_for_other_tests import (
    ensure_kill_all,
    kill_all_with_retries,
)
from truss.truss_handle import TrussHandle, wait_for_truss
from truss.types import Example


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
    resp = th.predict(
        {
            "inputs": [1, 2, 3, 4],
        }
    )
    assert resp == {"predictions": [4, 5, 6, 7]}


def test_predict_with_external_packages(custom_model_with_external_package):
    th = TrussHandle(custom_model_with_external_package)
    resp = th.predict(
        {
            "inputs": [1, 2, 3, 4],
        }
    )
    assert resp == [1, 1, 1, 1]


def test_server_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    resp = th.server_predict(
        {
            "inputs": [1, 2, 3, 4],
        }
    )
    assert resp == {"predictions": [4, 5, 6, 7]}


def test_readme_generation_int_example(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace("\n", "")
    correct_readme_contents = _read_readme("readme_int_example.md")
    assert readme_contents == correct_readme_contents


def test_readme_generation_no_example(
    custom_model_truss_dir_with_pre_and_post_no_example,
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post_no_example)
    # Remove the examples file
    os.remove(th._spec.examples_path)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace("\n", "")
    correct_readme_contents = _read_readme("readme_no_example.md")
    assert readme_contents == correct_readme_contents


def test_readme_generation_str_example(
    custom_model_truss_dir_with_pre_and_post_str_example,
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post_str_example)
    readme_contents = th.generate_readme()
    readme_contents = readme_contents.replace("\n", "")
    correct_readme_contents = _read_readme("readme_str_example.md")
    assert readme_contents == correct_readme_contents


@pytest.mark.integration
def test_build_docker_image(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-build-image-tag:0.0.1"
    image = th.build_serving_docker_image(tag=tag)
    assert image.repo_tags[0] == tag


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
    container = th.docker_run(tag=tag)
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.skip(reason="Needs gpu")
@pytest.mark.integration
def test_docker_run_gpu(custom_model_truss_dir_for_gpu):
    th = TrussHandle(custom_model_truss_dir_for_gpu)
    tag = "test-docker-run-gpu-tag:0.0.1"
    container = th.docker_run(tag=tag)
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def test_docker_run_without_tag(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    container = th.docker_run()
    try:
        assert _container_exists(container)
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def get_docker_containers_from_labels(custom_model_truss_dir_with_pre_and_post):
    with ensure_kill_all():
        t1 = TrussHandle(custom_model_truss_dir_with_pre_and_post)
        assert len(t1.get_serving_docker_containers_from_labels()) == 0
        t1.docker_run()
        assert len(t1.get_serving_docker_containers_from_labels()) == 1
        t1.docker_run(port=3000)
        assert len(t1.get_serving_docker_containers_from_labels()) == 2
        t1.kill_container()
        assert len(t1.get_serving_docker_containers_from_labels()) == 0


@pytest.mark.integration
def test_predict_use_docker(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-docker-predict-tag:0.0.1"
    with ensure_kill_all():
        result = th.predict({"inputs": [1, 2]}, tag=tag, use_docker=True)
        assert result == {"predictions": [4, 5]}


@pytest.mark.integration
def test_docker_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-docker-predict-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1, 2]}, tag=tag)
        assert result == {"predictions": [4, 5]}


@pytest.mark.integration
def test_docker_predict_model_with_external_packages(
    custom_model_with_external_package,
):
    th = TrussHandle(custom_model_with_external_package)
    tag = "test-docker-predict-ext-pkg-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1, 2]}, tag=tag)
        assert result == [1, 1]


@pytest.mark.integration
def test_docker_train(variables_to_artifacts_training_truss):
    th = TrussHandle(variables_to_artifacts_training_truss)
    th.add_training_variable("x", "y")
    th.add_training_variable("a", "b")
    tag = "test-docker-train-tag:0.0.1"
    with ensure_kill_all():
        input_vars = {"x": "z"}
        th.docker_train(variables=input_vars, tag=tag)
        vars_artifact = th.spec.data_dir / "variables.json"
        with vars_artifact.open() as vars_file:
            vars_from_artifact = json.load(vars_file)
            assert vars_from_artifact == {
                "x": "z",
                "a": "b",
            }


def test_local_train(variables_to_artifacts_training_truss):
    th = TrussHandle(variables_to_artifacts_training_truss)
    th.add_training_variable("x", "y")
    th.add_training_variable("a", "b")
    input_vars = {"x": "z"}
    th.local_train(variables=input_vars)
    vars_artifact = th.spec.data_dir / "variables.json"
    with vars_artifact.open() as vars_file:
        vars_from_artifact = json.load(vars_file)
        assert vars_from_artifact == {
            "x": "z",
            "a": "b",
        }


@pytest.mark.integration
def test_docker_predict_with_bundled_packages(
    custom_model_truss_dir_with_bundled_packages,
):
    th = TrussHandle(custom_model_truss_dir_with_bundled_packages)
    tag = "test-docker-predict-bundled-packages-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1, 2]}, tag=tag)
        assert result == {"predictions": [1]}


@pytest.mark.integration
def test_docker_multiple_predict(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    tag = "test-docker-predict-tag:0.0.1"
    with ensure_kill_all():
        r1 = th.docker_predict({"inputs": [1, 2]}, tag=tag)
        r2 = th.docker_predict({"inputs": [3, 4]}, tag=tag)
        assert r1 == {"predictions": [4, 5]}
        assert r2 == {"predictions": [6, 7]}
        assert len(th.get_serving_docker_containers_from_labels()) == 1


@pytest.mark.integration
def test_kill_all(custom_model_truss_dir, custom_model_truss_dir_with_pre_and_post):
    t1 = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    t2 = TrussHandle(custom_model_truss_dir)
    with ensure_kill_all():
        t1.docker_run()
        assert len(t1.get_serving_docker_containers_from_labels()) == 1
        t2.docker_run(local_port=3000)
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
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result["predictions"][0]["cuda_version"].startswith("11")


@pytest.mark.integration
def test_docker_predict_secrets(custom_model_truss_dir_for_secrets):
    th = TrussHandle(custom_model_truss_dir_for_secrets)
    tag = "test-docker-predict-secrets-tag:0.0.1"
    LocalConfigHandler.set_secret("secret_name", "secret_value")
    with ensure_kill_all():
        try:
            result = th.docker_predict({"inputs": ["secret_name"]}, tag=tag)
            assert result["predictions"][0] == "secret_value"
        finally:
            LocalConfigHandler.remove_secret("secret_name")


@pytest.mark.integration
def test_docker_no_preprocess_custom_model(no_preprocess_custom_model):
    th = TrussHandle(no_preprocess_custom_model)
    tag = "test-docker-no-preprocess-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result["predictions"][0] == 2


@pytest.mark.integration
def test_docker_long_load(long_load_model):
    th = TrussHandle(long_load_model)
    tag = "test-docker-long-load-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result["predictions"][0] == 1


@pytest.mark.integration
def test_local_no_preprocess_custom_model(no_preprocess_custom_model):
    th = TrussHandle(no_preprocess_custom_model)
    result = th.server_predict({"inputs": [1]})
    assert result["predictions"][0] == 2


@pytest.mark.integration
def test_docker_no_postprocess_custom_model(no_postprocess_custom_model):
    th = TrussHandle(no_postprocess_custom_model)
    tag = "test-docker-no-postprocess-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result["predictions"][0] == 2


@pytest.mark.integration
def test_local_no_postprocess_custom_model(no_postprocess_custom_model):
    th = TrussHandle(no_postprocess_custom_model)
    result = th.server_predict({"inputs": [1]})
    assert result["predictions"][0] == 2


@pytest.mark.integration
def test_docker_no_load_custom_model(no_load_custom_model):
    th = TrussHandle(no_load_custom_model)
    tag = "test-docker-no-load-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result["predictions"][0] == 1


@pytest.mark.integration
def test_local_no_load_custom_model(no_load_custom_model):
    th = TrussHandle(no_load_custom_model)
    result = th.server_predict({"inputs": [1]})
    assert result["predictions"][0] == 1


@pytest.mark.integration
def test_docker_no_params_init_custom_model(no_params_init_custom_model):
    th = TrussHandle(no_params_init_custom_model)
    tag = "test-docker-no-params-init-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result["predictions"][0] == 1


@pytest.mark.integration
def test_local_no_params_init_custom_model(no_params_init_custom_model):
    th = TrussHandle(no_params_init_custom_model)
    result = th.server_predict({"inputs": [1]})
    assert result["predictions"][0] == 1


@pytest.mark.integration
def test_custom_python_requirement(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_python_requirement("theano")
    th.add_python_requirement("scipy")
    tag = "test-custom-python-req-tag:0.0.1"
    container = th.docker_run(tag=tag)
    try:
        _verify_python_requirement_installed_on_container(container, "theano")
        _verify_python_requirement_installed_on_container(container, "scipy")
    finally:
        Docker.client().kill(container)


@pytest.mark.integration
def test_custom_system_package(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_system_package("jq")
    th.add_system_package("fzf")
    tag = "test-custom-system-package-tag:0.0.1"
    container = th.docker_run(tag=tag)
    try:
        _verify_system_package_installed_on_container(container, "jq")
        _verify_system_package_installed_on_container(container, "fzf")
    finally:
        Docker.client().kill(container)


def test_enable_gpu(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.enable_gpu()
    assert th.spec.config.resources.use_gpu


@pytest.mark.parametrize(
    "python_version, expected_python_version",
    [
        ("3.8", "py38"),
        ("py38", "py38"),
    ],
)
def test_update_python_version(
    python_version,
    expected_python_version,
    custom_model_truss_dir_with_pre_and_post,
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.update_python_version(python_version)
    assert th.spec.python_version == expected_python_version


def test_update_requirements(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
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
    allowed_requirements = [
        "tensorflow==2.3.1",
        "uvicorn==0.12.2",
    ]
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
    container = th.docker_run(tag=tag)
    try:
        _verify_environment_variable_on_container(container, "test_env", "test_value")
    finally:
        Docker.client().kill(container)


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


def test_example(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    assert "inputs" in th.example("example1").input


def test_example_index(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    assert "inputs" in th.example(0).input


def test_add_example_new(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    orig_examples = th.examples()
    th.add_example("example2", {"inputs": [[1]]})
    assert th.examples() == [
        *orig_examples,
        Example("example2", {"inputs": [[1]]}),
    ]


def test_add_example_update(custom_model_truss_dir_with_pre_and_post):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th.add_example("example1", {"inputs": [[1]]})
    assert th.examples() == [
        Example("example1", {"inputs": [[1]]}),
    ]


def test_model_without_pre_post(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    resp = th.server_predict(
        {
            "inputs": [1, 2, 3, 4],
        }
    )
    assert resp == [1, 1, 1, 1]


@pytest.mark.integration
def test_docker_predict_model_without_pre_post(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    with ensure_kill_all():
        resp = th.docker_predict(
            {
                "inputs": [1, 2, 3, 4],
            }
        )
        assert resp == [1, 1, 1, 1]


@pytest.mark.integration
def test_control_truss_apply_patch(custom_model_control):
    th = TrussHandle(custom_model_control)
    tag = "test-docker-custom-model-control-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result[0] == 1

        running_hash = th.truss_hash_on_serving_container()
        new_model_code = """
class Model:
    def predict(self, request):
        return [2 for i in request['inputs']]
"""
        patch_request = {
            "hash": "dummy",
            "prev_hash": running_hash,
            "patches": [
                Patch(
                    type=PatchType.MODEL_CODE,
                    body=ModelCodePatch(
                        action=Action.UPDATE,
                        path="model.py",
                        content=new_model_code,
                    ),
                ).to_dict(),
            ],
        }

        th.patch_container(patch_request)
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result[0] == 2


@pytest.mark.integration
def test_regular_truss_local_update_flow(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    tag = "test-docker-custom-model-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result[0] == 1
        orig_num_truss_images = len(th.get_all_docker_images())

        # No new docker images on second predict
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert orig_num_truss_images == len(th.get_all_docker_images())

        with (custom_model_truss_dir / "model" / "model.py").open(
            "w"
        ) as model_code_file:
            model_code_file.write(
                """
class Model:
    def predict(self, request):
        return [2 for i in request['inputs']]
"""
            )
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result[0] == 2
        # A new image should have been created
        assert len(th.get_all_docker_images()) == orig_num_truss_images + 1


@patch("truss.truss_handle.directory_content_hash")
def test_truss_hash_caching_based_on_max_mod_time(
    directory_content_patcher,
    custom_model_truss_dir,
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


@patch("truss.truss_handle.get_container_state")
def test_container_oom_caught_during_waiting(container_state_mock):
    container_state_mock.return_value = DockerStates.OOMKILLED
    with pytest.raises(ContainerIsDownError):
        wait_for_truss(url="localhost:8000", container=MagicMock())


@patch("truss.truss_handle.get_container_state")
@pytest.mark.integration
def test_container_stuck_in_created(container_state_mock):
    container_state_mock.return_value = DockerStates.CREATED
    with pytest.raises(ContainerIsDownError):
        wait_for_truss(url="localhost:8000", container=MagicMock())


@pytest.mark.integration
@pytest.mark.parametrize(
    "binary, python_version",
    [
        (binary, python_version)
        for binary in [True, False]
        for python_version in ["3.8", "3.9"]
    ],
)
def test_control_truss_local_update_flow(binary, python_version, custom_model_control):
    th = TrussHandle(custom_model_control)
    th.update_python_version(python_version)
    tag = "test-docker-custom-model-control-tag:0.0.1"

    def predict_with_updated_model_code():
        new_model_code = """
class Model:
    def predict(self, request):
        return [2 for i in request['inputs']]
"""
        model_code_file_path = custom_model_control / "model" / "model.py"
        with model_code_file_path.open("w") as model_code_file:
            model_code_file.write(new_model_code)
        return th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)

    def predict_with_added_empty_directory():
        # Adding empty directory should work
        (custom_model_control / "model" / "dir").mkdir()
        return th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)

    def predict_with_unpatchable_change():
        # Changes that are not expressible with patch should also work
        # Changes to data dir are not currently patch expressible
        (custom_model_control / "data" / "dummy").touch()
        return th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)

    def predict_with_python_requirement_added(req: str):
        th.add_python_requirement(req)
        return th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)

    def predict_with_python_requirement_removed(req):
        th.remove_python_requirement(req)
        return th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)

    def predict_with_system_requirement_added(pkg):
        th.add_system_package(pkg)
        return th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)

    def predict_with_system_requirement_removed(pkg):
        th.remove_system_package(pkg)
        return th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)

    def current_num_docker_images() -> int:
        return len(th.get_all_docker_images())

    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag, binary=binary)
        assert result[0] == 1
        orig_num_truss_images = len(th.get_all_docker_images())

        result = predict_with_updated_model_code()
        assert result[0] == 2
        assert orig_num_truss_images == current_num_docker_images()

        result = predict_with_added_empty_directory()
        assert result[0] == 2
        assert orig_num_truss_images == current_num_docker_images()

        container = th._get_running_serving_container_ignore_hash()

        python_req = "pydot"
        result = predict_with_python_requirement_added(python_req)
        assert result[0] == 2
        assert current_num_docker_images() == orig_num_truss_images
        _verify_python_requirement_installed_on_container(container, python_req)

        result = predict_with_python_requirement_removed(python_req)
        assert result[0] == 2
        assert current_num_docker_images() == orig_num_truss_images
        _verify_python_requirement_not_installed_on_container(container, python_req)

        system_pkg = "jq"
        result = predict_with_system_requirement_added(system_pkg)
        assert result[0] == 2
        assert current_num_docker_images() == orig_num_truss_images
        _verify_system_package_installed_on_container(container, system_pkg)

        result = predict_with_system_requirement_removed(system_pkg)
        assert result[0] == 2
        assert current_num_docker_images() == orig_num_truss_images
        _verify_system_requirement_not_installed_on_container(container, system_pkg)

        result = predict_with_unpatchable_change()
        assert result[0] == 2
        assert current_num_docker_images() == orig_num_truss_images + 1


@pytest.mark.integration
def test_control_truss_huggingface(
    huggingface_truss_handle_small_model,
):
    th = TrussHandle(huggingface_truss_handle_small_model)
    th.live_reload()
    tag = "test-docker-huggingface-model-control-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict(
            {
                "inputs": ["My name is Sarah and I live in London"],
            },
            tag=tag,
        )
        predictions = result["predictions"]
        assert len(predictions) == 1
        prediction = predictions[0]
        assert prediction["generated_text"].startswith("Mein Name")


@pytest.mark.integration
def test_control_truss_local_update_that_crashes_inference_server(custom_model_control):
    th = TrussHandle(custom_model_control)
    tag = "test-docker-custom-model-control-tag:0.0.1"
    with ensure_kill_all():
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result[0] == 1

        bad_model_code = """
class Model:
    def malformed
"""
        model_code_file_path = custom_model_control / "model" / "model.py"
        with model_code_file_path.open("w") as model_code_file:
            model_code_file.write(bad_model_code)
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            th.docker_predict({"inputs": [1]}, tag=tag)
        resp = exc_info.value.response
        assert resp.status_code == 500
        assert "Model load failed" in resp.text

        # Should be able to fix code after
        good_model_code = """
class Model:
    def predict(self, request):
        return [2 for i in request['inputs']]
"""
        with model_code_file_path.open("w") as model_code_file:
            model_code_file.write(good_model_code)
        result = th.docker_predict({"inputs": [1]}, tag=tag)
        assert result[0] == 2


@pytest.mark.integration
@pytest.mark.parametrize(
    "patch_path, expected_call_count",
    [
        ("hash_is_current", 1),
        ("hash_is_current_but_only_every_third_call_succeeds", 3),
    ],
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
            {"inputs": [1]},
            tag=tag,
            patch_ping_url=patch_ping_url,
        )
        assert result == [1]

        # Make sure the patch ping url was actually hit
        stats = requests.get(f"http://127.0.0.1:{port}/stats").json()
        assert stats[f"{patch_path}_called_count"] == expected_call_count


def test_handle_if_container_dne(custom_model_truss_dir):
    def return_container_dne(self):
        return "DNE"

    with patch.object(
        TrussHandle, "_try_patch", new=return_container_dne
    ), pytest.raises(ContainerNotFoundError):
        truss_handle = TrussHandle(truss_dir=custom_model_truss_dir)
        truss_handle.docker_run(local_port=3000)
    kill_all_with_retries()


def test_docker_predict_container_does_not_exist(custom_model_truss_dir):
    def return_container_dne(self):
        return "DNE"

    with patch.object(
        TrussHandle, "_try_patch", new=return_container_dne
    ), pytest.raises(ContainerNotFoundError):
        truss_handle = TrussHandle(truss_dir=custom_model_truss_dir)
        truss_handle.docker_predict({"inputs": [1]}, local_port=3000)
    kill_all_with_retries()


def _container_exists(container) -> bool:
    for row in Docker.client().ps():
        if row.id.startswith(container.id):
            return True
    return False


def _verify_system_package_installed_on_container(container, pkg: str):
    resp = container.execute(["which", pkg])
    assert resp.strip() == f"/usr/bin/{pkg}"


def _verify_system_requirement_not_installed_on_container(container, pkg: str):
    try:
        container.execute(["dpkg", "-l", pkg])
    except DockerException as excp:
        assert "no packages found" in str(excp)


def _verify_python_requirement_installed_on_container(container, req: str):
    resp = container.execute(["pip", "show", req])
    assert resp.splitlines()[0].lower() == f"Name: {req}".lower()


def _verify_python_requirement_not_installed_on_container(container, req: str):
    try:
        container.execute(["pip", "show", req])
    except DockerException as excp:
        assert "not found" in str(excp)


def _verify_environment_variable_on_container(
    container,
    env_var_name: str,
    env_var_value: str,
):
    resp = container.execute(["env"])
    needle = f"{env_var_name}={env_var_value}"
    assert needle in resp.splitlines()


def _read_readme(filename: str) -> str:
    readme_correct_path = Path(__file__).parent.parent / "test_data" / filename
    readme_contents = readme_correct_path.open().read().replace("\n", "")
    return readme_contents
