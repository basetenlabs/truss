import os
from pathlib import Path
from typing import Any, Callable, List, Optional

import pytest
import yaml
from truss.patch.calc_patch import calc_truss_patch, calc_unignored_paths
from truss.patch.signature import calc_truss_signature
from truss.templates.control.control.helpers.types import (
    Action,
    ConfigPatch,
    EnvVarPatch,
    ExternalDataPatch,
    ModelCodePatch,
    PackagePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.truss_config import TrussConfig
from truss.truss_handle import TrussHandle


def test_calc_truss_patch_unsupported(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)

    # Unsupported directory should result in no patches
    (custom_model_truss_dir / "data").touch()
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)
    assert len(patches) == 0

    # Changes under unsupported directory should return None to reflect
    # inability to calculate patch.
    (custom_model_truss_dir / "data" / "dummy").touch()
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)
    assert patches is None


def test_calc_truss_patch_add_file(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    with (custom_model_truss_dir / "model" / "dummy").open("w") as file:
        file.write("content")
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.ADD,
            path="dummy",
            content="content",
        ),
    )


def test_calc_truss_patch_add_under_new_directory(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    new_dir = custom_model_truss_dir / "model" / "dir"
    new_dir.mkdir()
    (new_dir / "dummy").touch()
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.ADD,
            path="dir/dummy",
            content="",
        ),
    )


def test_calc_truss_patch_remove_file(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    (custom_model_truss_dir / "model" / "model.py").unlink()
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.REMOVE,
            path="model.py",
        ),
    )


def test_calc_truss_patch_update_file(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    new_model_file_content = """
    class Model:
        pass
    """
    with (custom_model_truss_dir / "model" / "model.py").open("w") as model_file:
        model_file.write(new_model_file_content)

    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE, path="model.py", content=new_model_file_content
        ),
    )


def test_calc_truss_ignore_pycache(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    top_pycache_path = custom_model_truss_dir / "__pycache__"
    top_pycache_path.mkdir()
    (top_pycache_path / "bla.pyc").touch()
    model_pycache_path = custom_model_truss_dir / "model" / "__pycache__"
    model_pycache_path.mkdir()
    (model_pycache_path / "foo.pyo").touch()

    patches = calc_truss_patch(
        custom_model_truss_dir,
        prev_sign,
    )
    assert len(patches) == 0


def test_calc_truss_ignore_pycache_existing(custom_model_truss_dir: Path):
    # If __pycache__ existed before and there are no changes, there should be no
    # patches.
    top_pycache_path = custom_model_truss_dir / "__pycache__"
    top_pycache_path.mkdir()
    (top_pycache_path / "bla.pyc").touch()
    model_pycache_path = custom_model_truss_dir / "model" / "__pycache__"
    model_pycache_path.mkdir()
    (model_pycache_path / "foo.pyo").touch()
    sign = calc_truss_signature(custom_model_truss_dir)
    patches = calc_truss_patch(
        custom_model_truss_dir,
        sign,
    )
    assert len(patches) == 0


def test_calc_truss_ignore_changes_outside_patch_relevant_dirs(
    custom_model_truss_dir: Path,
):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    top_pycache_path = custom_model_truss_dir / "__pycache__"
    top_pycache_path.mkdir()
    (top_pycache_path / "README.md").touch()
    git_dir = custom_model_truss_dir / ".git"
    git_dir.mkdir()
    (git_dir / "dummy").touch()

    patches = calc_truss_patch(
        custom_model_truss_dir,
        prev_sign,
    )
    assert len(patches) == 0

    # Removing should also be ignored
    new_sign = calc_truss_signature(custom_model_truss_dir)
    (git_dir / "dummy").unlink()
    patches = calc_truss_patch(
        custom_model_truss_dir,
        new_sign,
    )
    assert len(patches) == 0


def test_calc_config_patches_add_python_requirement(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        lambda config: config.requirements.append("requests==1.0.0"),
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.ADD,
                requirement="requests==1.0.0",
            ),
        ),
    ]


def test_calc_truss_patch_add_package(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    new_dir = custom_model_truss_dir / "packages" / "dir"
    new_dir.mkdir()
    (new_dir / "dummy").touch()
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(
            action=Action.ADD,
            path="dir/dummy",
            content="",
        ),
    )


def test_calc_truss_patch_remove_package(
    custom_model_truss_dir_with_bundled_packages: Path,
):
    prev_sign = calc_truss_signature(custom_model_truss_dir_with_bundled_packages)
    (
        custom_model_truss_dir_with_bundled_packages
        / "packages"
        / "test_package"
        / "test.py"
    ).unlink()
    patches = calc_truss_patch(custom_model_truss_dir_with_bundled_packages, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(
            action=Action.REMOVE,
            path="test_package/test.py",
        ),
    )


def test_calc_truss_patch_update_package(
    custom_model_truss_dir_with_bundled_packages: Path,
):
    prev_sign = calc_truss_signature(custom_model_truss_dir_with_bundled_packages)
    new_package_file_content = """X = 2"""
    with (
        custom_model_truss_dir_with_bundled_packages
        / "packages"
        / "test_package"
        / "test.py"
    ).open("w") as package_file:
        package_file.write(new_package_file_content)

    patches = calc_truss_patch(custom_model_truss_dir_with_bundled_packages, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(
            action=Action.UPDATE,
            path="test_package/test.py",
            content=new_package_file_content,
        ),
    )


def test_calc_truss_patch_handles_requirements_file_name_change(
    custom_model_truss_dir: Path,
):
    requirements_contents = """xformers"""

    def pre_config_op(config: TrussConfig):
        filename = "requirement.txt"
        config.requirements.clear()
        config.requirements_file = filename
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)

    def config_op(config: TrussConfig):
        filename = "requirements.txt"
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)
        config.requirements_file = filename
        config.requirements.clear()

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_op=config_op,
        config_pre_op=pre_config_op,
    )
    assert len(patches) == 1
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
    ]


def test_calc_truss_patch_handles_requirements_file_changes(
    custom_model_truss_dir: Path,
):
    def pre_config_op(config: TrussConfig):
        requirements_contents = """xformers\ntorch==2.0.1"""
        filename = "./requirements.txt"
        config.requirements_file = filename
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)

    def config_op(config: TrussConfig):
        requirements_contents = """requests\ntorch==2.3.1"""
        filename = "requirements.txt"
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_op=config_op,
        config_pre_op=pre_config_op,
    )
    assert len(patches) == 3
    assert patches == [
        # In this case, a Config Update patch is not issued. This does not cause issues on the backend
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.REMOVE,
                requirement="xformers",
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.ADD,
                requirement="requests",
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.UPDATE,
                requirement="torch==2.3.1",
            ),
        ),
    ]


def test_calc_truss_patch_handles_requirements_file_changes_and_config_changes(
    custom_model_truss_dir: Path,
):
    def pre_config_op(config: TrussConfig):
        requirements_contents = """xformers\ntorch==2.0.1"""
        filename = "./requirements.txt"
        config.requirements_file = filename
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)

    def config_op(config: TrussConfig):
        requirements_contents = """requests\ntorch==2.3.1"""
        filename = "requirement.txt"
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)
        config.requirements_file = filename

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_op=config_op,
        config_pre_op=pre_config_op,
    )
    assert len(patches) == 4
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.REMOVE,
                requirement="xformers",
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.ADD,
                requirement="requests",
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.UPDATE,
                requirement="torch==2.3.1",
            ),
        ),
    ]


def test_calc_truss_patch_handles_requirements_file_removal(
    custom_model_truss_dir: Path,
):
    requirements_contents = """xformers"""
    filename = "requirements.txt"

    def pre_config_op(config: TrussConfig):
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)
        config.requirements_file = filename
        config.requirements.clear()

    def config_op(config: TrussConfig):
        requirements_contents = ["xformers", "requests"]
        config.requirements.extend(requirements_contents)
        config.requirements_file = ""
        os.remove(custom_model_truss_dir / filename)

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_op=config_op,
        config_pre_op=pre_config_op,
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.ADD,
                requirement="requests",
            ),
        ),
    ]


def test_calc_truss_signature_raises_for_invalid_requirements_file(
    custom_model_truss_dir: Path,
):
    config_path = custom_model_truss_dir / "config.yaml"
    config = TrussConfig.from_yaml(config_path)
    config.requirements.clear()
    config.requirements_file = "no_exist.txt"
    config.write_to_yaml_file(config_path)

    with pytest.raises(FileNotFoundError):
        calc_truss_signature(custom_model_truss_dir)


def test_calc_truss_patch_handles_requirements_file_added_no_change(
    custom_model_truss_dir: Path,
):
    requirements_contents = """xformers"""

    def pre_config_op(config: TrussConfig):
        config.requirements.append(requirements_contents)

    def config_op(config: TrussConfig):
        filename = "requirements.txt"
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)
        config.requirements_file = filename
        config.requirements.clear()

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_op=config_op,
        config_pre_op=pre_config_op,
    )
    assert len(patches) == 1
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
    ]


def test_calc_truss_patch_handles_requirements_file_added_with_changes(
    custom_model_truss_dir: Path,
):
    requirements_contents = """xformers"""

    def config_op(config: TrussConfig):
        filename = "requirements.txt"
        with (custom_model_truss_dir / filename).open("w") as req_file:
            req_file.write(requirements_contents)
        config.requirements_file = filename
        config.requirements.clear()

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir, config_op=config_op
    )

    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.ADD,
                requirement="xformers",
            ),
        ),
    ]


def test_calc_config_patches_remove_python_requirement(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=lambda config: config.requirements.append("requests==1.0.0"),
        config_op=lambda config: config.requirements.clear(),
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.REMOVE,
                requirement="requests",
            ),
        ),
    ]


def test_calc_config_patches_update_python_requirement(custom_model_truss_dir: Path):
    def update_requests_version(config: TrussConfig):
        config.requirements[0] = "requests==2.0.0"

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=lambda config: config.requirements.append("requests==1.0.0"),
        config_op=update_requests_version,
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.UPDATE,
                requirement="requests==2.0.0",
            ),
        ),
    ]


def test_calc_config_patches_add_remove_and_update_python_requirement(
    custom_model_truss_dir: Path,
):
    def config_pre_op(config: TrussConfig):
        config.requirements = [
            "requests==1.0.0",
            "jinja==4.0.0",
        ]

    def config_op(config: TrussConfig):
        config.requirements = [
            "requests==2.0.0",
            "numpy>=1.8",
        ]

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=config_pre_op,
        config_op=config_op,
    )
    assert len(patches) == 4
    assert patches[0] == Patch(
        type=PatchType.CONFIG,
        body=ConfigPatch(
            action=Action.UPDATE,
            config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
        ),
    )
    patches = patches[1:]
    patches.sort(key=lambda patch: patch.body.requirement)
    assert patches == [
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.REMOVE,
                requirement="jinja",
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.ADD,
                requirement="numpy>=1.8",
            ),
        ),
        Patch(
            type=PatchType.PYTHON_REQUIREMENT,
            body=PythonRequirementPatch(
                action=Action.UPDATE,
                requirement="requests==2.0.0",
            ),
        ),
    ]


def test_calc_config_patches_add_env_var(
    custom_model_truss_dir: Path,
):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_op=lambda config: config.environment_variables.update({"foo": "bar"}),
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.ENVIRONMENT_VARIABLE,
            body=EnvVarPatch(
                action=Action.ADD,
                item={"foo": "bar"},
            ),
        ),
    ]


def test_calc_config_patches_add_remove_env_var(
    custom_model_truss_dir: Path,
):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=lambda config: config.environment_variables.update(
            {"foo": "bar"}
        ),
        config_op=lambda config: config.environment_variables.clear(),
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.ENVIRONMENT_VARIABLE,
            body=EnvVarPatch(
                action=Action.REMOVE,
                item={"foo": "bar"},
            ),
        ),
    ]


def test_calc_config_patches_add_system_package(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        lambda config: config.system_packages.append("curl"),
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.SYSTEM_PACKAGE,
            body=SystemPackagePatch(
                action=Action.ADD,
                package="curl",
            ),
        ),
    ]


def test_calc_config_patches_remove_system_package(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=lambda config: config.system_packages.append("curl"),
        config_op=lambda config: config.system_packages.clear(),
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.SYSTEM_PACKAGE,
            body=SystemPackagePatch(
                action=Action.REMOVE,
                package="curl",
            ),
        ),
    ]


def test_calc_config_patches_add_and_remove_system_package(
    custom_model_truss_dir: Path,
):
    def config_pre_op(config: TrussConfig):
        config.system_packages = [
            "curl",
            "jq",
        ]

    def config_op(config: TrussConfig):
        config.system_packages = [
            "curl",
            "libsnd",
        ]

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=config_pre_op,
        config_op=config_op,
    )
    assert len(patches) == 3
    assert patches[0] == Patch(
        type=PatchType.CONFIG,
        body=ConfigPatch(
            action=Action.UPDATE,
            config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
        ),
    )
    patches = patches[1:]
    patches.sort(key=lambda patch: patch.body.package)
    assert patches == [
        Patch(
            type=PatchType.SYSTEM_PACKAGE,
            body=SystemPackagePatch(
                action=Action.REMOVE,
                package="jq",
            ),
        ),
        Patch(
            type=PatchType.SYSTEM_PACKAGE,
            body=SystemPackagePatch(
                action=Action.ADD,
                package="libsnd",
            ),
        ),
    ]


def test_calc_config_patches_toggle_apply_library_patches(custom_model_truss_dir: Path):
    def config_op(config: TrussConfig):
        config.apply_library_patches = False

    patches = _apply_config_change_and_calc_patches(custom_model_truss_dir, config_op)
    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.CONFIG,
        body=ConfigPatch(
            action=Action.UPDATE,
            config=yaml.safe_load((custom_model_truss_dir / "config.yaml").open()),
        ),
    )


def test_calc_config_patches_add_external_data(
    custom_model_external_data_access_tuple_fixture: Path,
):
    path, _ = custom_model_external_data_access_tuple_fixture
    th = TrussHandle(path)
    external_data = th.spec.config.external_data

    def config_op(config: TrussConfig):
        config.external_data = external_data

    def config_pre_op(config: TrussConfig):
        config.external_data = None

    patches = _apply_config_change_and_calc_patches(
        path,
        config_pre_op=config_pre_op,
        config_op=config_op,
    )
    assert len(patches) == 2

    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((path / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.EXTERNAL_DATA,
            body=ExternalDataPatch(action=Action.ADD, item=external_data.to_list()[0]),
        ),
    ]


def test_calc_config_patches_remove_external_data(
    custom_model_external_data_access_tuple_fixture: Path,
):
    path, _ = custom_model_external_data_access_tuple_fixture
    th = TrussHandle(path)
    external_data = th.spec.config.external_data

    def config_op(config: TrussConfig):
        config.external_data = None

    patches = _apply_config_change_and_calc_patches(
        path,
        config_op=config_op,
    )
    assert len(patches) == 2
    assert patches == [
        Patch(
            type=PatchType.CONFIG,
            body=ConfigPatch(
                action=Action.UPDATE,
                config=yaml.safe_load((path / "config.yaml").open()),
            ),
        ),
        Patch(
            type=PatchType.EXTERNAL_DATA,
            body=ExternalDataPatch(
                action=Action.REMOVE, item=external_data.to_list()[0]
            ),
        ),
    ]


def test_calc_unignored_paths():
    ignore_patterns = [
        ".mypy_cache/",
        "venv/",
        "*.tmp",
    ]

    root_relative_paths = {
        ".mypy_cache/should_ignore.json",
        "venv/bin/activate",
        "ignored_file.tmp",
        "config.yaml",
        "model/model.py",
    }

    unignored_paths = calc_unignored_paths(root_relative_paths, ignore_patterns)
    assert unignored_paths == {
        "config.yaml",
        "model/model.py",
    }


def _apply_config_change_and_calc_patches(
    custom_model_truss_dir: Path,
    config_op: Callable[[TrussConfig], Any],
    config_pre_op: Optional[Callable[[TrussConfig], Any]] = None,
) -> List[Patch]:
    def modify_config(op):
        config_path = custom_model_truss_dir / "config.yaml"
        config = TrussConfig.from_yaml(config_path)
        op(config)
        config.write_to_yaml_file(config_path)

    if config_pre_op is not None:
        modify_config(config_pre_op)

    prev_sign = calc_truss_signature(custom_model_truss_dir)
    modify_config(config_op)
    return calc_truss_patch(custom_model_truss_dir, prev_sign)
