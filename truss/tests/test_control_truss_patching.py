from pathlib import Path

import pytest

from truss.base.constants import SUPPORTED_PYTHON_VERSIONS
from truss.base.truss_config import ExternalData, ExternalDataItem
from truss.local.local_config_handler import LocalConfigHandler
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.tests.test_truss_handle import (
    verify_python_requirement_installed_on_container,
    verify_python_requirement_not_installed_on_container,
    verify_system_package_installed_on_container,
    verify_system_requirement_not_installed_on_container,
)
from truss.truss_handle.truss_gatherer import calc_shadow_truss_dirname
from truss.truss_handle.truss_handle import TrussHandle


def current_num_docker_images(th: TrussHandle) -> int:
    return len(th.get_all_docker_images())


@pytest.fixture
def control_model_handle_tag_tuple(
    custom_model_control,
) -> tuple[Path, TrussHandle, str]:
    th = TrussHandle(custom_model_control)
    tag = "test-docker-custom-model-control-tag:0.0.1"
    return (custom_model_control, th, tag)


def update_model_code(path, new_model_code):
    model_code_file_path = path / "model" / "model.py"
    with model_code_file_path.open("w") as model_code_file:
        model_code_file.write(new_model_code)


@pytest.mark.integration
@pytest.mark.parametrize(
    "binary, python_version",
    [
        (binary, python_version)
        for binary in [True, False]
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ],
)
def test_control_truss_model_code_patch(
    binary, python_version, control_model_handle_tag_tuple
):
    custom_model_control, th, tag = control_model_handle_tag_tuple
    th.update_python_version(python_version)

    new_model_code = """
class Model:
    def predict(self, model_input):
        return [2 for i in model_input]
"""

    with ensure_kill_all():
        result = th.docker_predict([1], tag=tag, binary=binary, local_port=None)
        assert result[0] == 1
        orig_num_truss_images = len(th.get_all_docker_images())
        update_model_code(custom_model_control, new_model_code)
        result = th.docker_predict([1], tag=tag, binary=binary, local_port=None)
        assert result[0] == 2
        assert orig_num_truss_images == current_num_docker_images(th)


@pytest.mark.integration
@pytest.mark.parametrize(
    "binary, python_version",
    [
        (binary, python_version)
        for binary in [True, False]
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ],
)
def test_control_truss_empty_dir_patch(
    binary, python_version, control_model_handle_tag_tuple
):
    custom_model_control, th, tag = control_model_handle_tag_tuple
    th.update_python_version(python_version)

    def predict_with_added_empty_directory():
        # Adding empty directory should work
        (custom_model_control / "model" / "dir").mkdir()
        return th.docker_predict([1], tag=tag, binary=binary, local_port=None)

    with ensure_kill_all():
        th.docker_predict([1], tag=tag, binary=binary, local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())

        predict_with_added_empty_directory()
        assert orig_num_truss_images == current_num_docker_images(th)


# todo(justin): remove once this patch is supported
@pytest.mark.integration
@pytest.mark.parametrize(
    "binary, python_version",
    [
        (binary, python_version)
        for binary in [True, False]
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ],
)
def test_control_truss_unpatchable(
    binary, python_version, control_model_handle_tag_tuple
):
    custom_model_control, th, tag = control_model_handle_tag_tuple
    th.update_python_version(python_version)

    def predict_with_unpatchable_change():
        # Changes that are not expressible with patch should also work
        # Changes to data dir are not currently patch expressible
        (custom_model_control / "data" / "dummy").touch()
        return th.docker_predict([1], tag=tag, binary=binary, local_port=None)

    with ensure_kill_all():
        th.docker_predict([1], tag=tag, binary=binary, local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())

        predict_with_unpatchable_change()
        assert current_num_docker_images(th) == orig_num_truss_images + 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "binary, python_version",
    [
        (binary, python_version)
        for binary in [True, False]
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ],
)
def test_control_truss_python_sys_req_patch(
    binary, python_version, control_model_handle_tag_tuple
):
    _, th, tag = control_model_handle_tag_tuple
    th.update_python_version(python_version)

    def predict_with_python_requirement_added(req: str):
        th.add_python_requirement(req)
        return th.docker_predict([1], tag=tag, binary=binary, local_port=None)

    def predict_with_python_requirement_removed(req):
        th.remove_python_requirement(req)
        return th.docker_predict([1], tag=tag, binary=binary, local_port=None)

    def predict_with_system_requirement_added(pkg):
        th.add_system_package(pkg)
        return th.docker_predict([1], tag=tag, binary=binary, local_port=None)

    def predict_with_system_requirement_removed(pkg):
        th.remove_system_package(pkg)
        return th.docker_predict([1], tag=tag, binary=binary, local_port=None)

    with ensure_kill_all():
        th.docker_predict([1], tag=tag, binary=binary, local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())

        container = th.get_running_serving_container_ignore_hash()

        python_req = "pydot"
        predict_with_python_requirement_added(python_req)
        assert current_num_docker_images(th) == orig_num_truss_images
        verify_python_requirement_installed_on_container(container, python_req)

        predict_with_python_requirement_removed(python_req)
        assert current_num_docker_images(th) == orig_num_truss_images
        verify_python_requirement_not_installed_on_container(container, python_req)

        system_pkg = "jq"
        predict_with_system_requirement_added(system_pkg)
        assert current_num_docker_images(th) == orig_num_truss_images
        verify_system_package_installed_on_container(container, system_pkg)

        predict_with_system_requirement_removed(system_pkg)
        assert current_num_docker_images(th) == orig_num_truss_images
        verify_system_requirement_not_installed_on_container(container, system_pkg)


@pytest.mark.integration
@pytest.mark.parametrize(
    "binary, python_version",
    [
        (binary, python_version)
        for binary in [True, False]
        for python_version in SUPPORTED_PYTHON_VERSIONS
    ],
)
def test_control_truss_patch_ignored_changes(
    binary, python_version, control_model_handle_tag_tuple
):
    custom_model_control, th, tag = control_model_handle_tag_tuple
    th.update_python_version(python_version)

    def predict_with_ignored_changes():
        top_pycache_path = custom_model_control / "__pycache__"
        top_pycache_path.mkdir()
        (top_pycache_path / "bla.pyc").touch()
        model_pycache_path = custom_model_control / "model" / "__pycache__"
        model_pycache_path.mkdir()
        (model_pycache_path / "foo.pyc").touch()
        return th.docker_predict([1], tag=tag, binary=binary, local_port=None)

    with ensure_kill_all():
        th.docker_predict([1], tag=tag, binary=binary, local_port=None)
        orig_num_truss_images = current_num_docker_images(th)

        predict_with_ignored_changes()
        assert current_num_docker_images(th) == orig_num_truss_images


@pytest.mark.integration
def test_patch_added_model_dir(
    custom_model_control, tmp_path, control_model_handle_tag_tuple
):
    custom_model_control, th, tag = control_model_handle_tag_tuple

    def predict_with_added_model_dir_file():
        code_file_dir = custom_model_control / "model" / "dir"
        code_file_dir.mkdir(parents=True)
        with (code_file_dir / "foo.bar").open("w") as model_code_file:
            model_code_file.write("foobar")
        return th.docker_predict([1], build_dir=tmp_path, tag=tag, local_port=None)

    with ensure_kill_all():
        th.docker_predict([1], tag=tag, local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())

        predict_with_added_model_dir_file()
        assert orig_num_truss_images == current_num_docker_images(th)
        assert (custom_model_control / "model" / "dir" / "foo.bar").exists()


# todo(abu/justin) remove once ignored
@pytest.mark.skip(reason="Unsupported patch")
@pytest.mark.integration
def test_patch_data_dir(control_model_handle_tag_tuple):
    custom_model_control, th, tag = control_model_handle_tag_tuple

    def predict_with_data_dir_change():
        path = custom_model_control / "data" / "dummy"
        path.touch()
        th.docker_predict([1], tag=tag, local_port=None)
        with path.open("w") as file:
            file.write("foobar")
        th.docker_predict([1], tag=tag, local_port=None)
        path.unlink()
        return th.docker_predict([1], tag=tag, local_port=None)

    with ensure_kill_all():
        th.docker_predict([1], tag=tag, local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())

        predict_with_data_dir_change()
        assert orig_num_truss_images == current_num_docker_images(th)


@pytest.mark.integration
def test_patch_env_var(control_model_handle_tag_tuple):
    truss_dir, th, tag = control_model_handle_tag_tuple

    def predict_with_environment_variables_change():
        th.add_environment_variable("foo", "bar")
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result == "bar"
        th.add_environment_variable("foo", "bar2")
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result == "bar2"
        config_path = truss_dir / "config.yaml"
        with config_path.open("w") as file:
            file.write("{}")

        assert th._serving_hash() != th.truss_hash_on_serving_container()
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result == "No foo :("

    with ensure_kill_all():
        new_model_code = """
import os
class Model:
    def predict(self, model_input):
        print(os.environ.copy())
        return os.environ.get("foo", "No foo :(")
"""
        update_model_code(truss_dir, new_model_code)
        result = th.docker_predict([1], tag=tag, local_port=None)
        assert result == "No foo :("
        orig_num_truss_images = len(th.get_all_docker_images())

        predict_with_environment_variables_change()
        assert orig_num_truss_images == current_num_docker_images(th)


@pytest.mark.integration
def test_patch_external_package_dirs(custom_model_with_external_package):
    th = TrussHandle(custom_model_with_external_package)
    tag = "test-docker-custom-model-control-external-package-tag:0.0.1"
    th.live_reload()
    with ensure_kill_all():
        th.docker_predict([1], tag=tag, local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())
        th.clear_external_packages()
        th.add_external_package("../ext_pkg_patched")
        th.add_external_package("../ext_pkg2_patched")
        ext_pkg_path = custom_model_with_external_package / ".." / "ext_pkg_patched"
        ext_pkg_path.mkdir()
        (ext_pkg_path / "subdir_patched").mkdir()
        (ext_pkg_path / "subdir_patched" / "sub_module.py").touch()
        (ext_pkg_path / "top_module.py").touch()
        ext_pkg_path2 = custom_model_with_external_package / ".." / "ext_pkg2_patched"
        ext_pkg_path2.mkdir()
        (ext_pkg_path2 / "top_module_patched.py").touch()
        new_model_code = """
import top_module
import subdir_patched.sub_module
import top_module_patched
class Model:
    def predict(self, model_input):
        return [1 for i in model_input]
"""
        update_model_code(custom_model_with_external_package, new_model_code)
        th.docker_predict([1], tag=tag, local_port=None)
        assert orig_num_truss_images == current_num_docker_images(th)
        shadow_truss_path = (
            LocalConfigHandler.shadow_trusses_dir_path()
            / calc_shadow_truss_dirname(custom_model_with_external_package)
        )
        packages_dir_path_in_shadow = (
            shadow_truss_path / th.spec.config.bundled_packages_dir
        )
        assert (packages_dir_path_in_shadow / "subdir_patched").exists() and (
            packages_dir_path_in_shadow / "top_module_patched.py"
        ).exists()


# todo(abu/justin) remove once ignored
@pytest.mark.skip(reason="Unsupported patch")
@pytest.mark.integration
def test_patch_secrets(control_model_handle_tag_tuple):
    _, th, tag = control_model_handle_tag_tuple

    def predict_with_secrets():
        th.add_secret("foo", "bar")
        return th.docker_predict([1], tag=tag, local_port=None)

    with ensure_kill_all():
        th.docker_predict([1], tag=tag, local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())

        predict_with_secrets()
        assert orig_num_truss_images == current_num_docker_images(th)


@pytest.mark.integration
def test_predict_with_external_data_change(
    custom_model_external_data_access_tuple_fixture, tmp_path
):
    truss_dir, _ = custom_model_external_data_access_tuple_fixture
    th = TrussHandle(truss_dir)
    th.live_reload()
    tag = "test-external-data-access-tag:0.0.1"
    with ensure_kill_all():
        th.docker_predict([], tag=tag, network="host", local_port=None)
        orig_num_truss_images = len(th.get_all_docker_images())
        th.remove_all_external_data()
        assert th._serving_hash() != th.truss_hash_on_serving_container()
        new_model_code = """
class Model:
    def __init__(self, data_dir):
        self._data_dir = data_dir
        pass

    def predict(self, model_input):
        return None
"""
        update_model_code(truss_dir, new_model_code)
        th.docker_predict([], tag=tag, network="host", local_port=None)
        content = "foobar"
        filename = "foobar.txt"
        (tmp_path / filename).write_text(content)
        new_model_code = """
class Model:
    def __init__(self, data_dir):
        self._data_dir = data_dir
        pass

    def predict(self, model_input):
        return (self._data_dir / 'foobar.txt').read_text()
"""
        update_model_code(truss_dir, new_model_code)
        url = f"http://host.docker.internal:9089/{filename}"
        th.add_external_data_item(url, filename)
        result = th.docker_predict([], tag=tag, network="host", local_port=None)
        assert result == content

        content = "patched content"
        new_filename = "foobar-patched.txt"
        (tmp_path / new_filename).write_text(content)
        new_external_data = ExternalData(
            [
                ExternalDataItem(
                    url=f"http://host.docker.internal:9089/{new_filename}",
                    local_data_path=filename,
                )
            ]
        )
        th._update_config(external_data=new_external_data)
        result = th.docker_predict([], tag=tag, network="host", local_port=None)
        assert orig_num_truss_images == current_num_docker_images(th)
        assert result == content
