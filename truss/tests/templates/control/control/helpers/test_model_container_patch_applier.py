import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from truss.base.truss_config import TrussConfig

# Needed to simulate the set up on the model docker container
sys.path.append(
    str(
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "templates"
        / "control"
        / "control"
    )
)

# Have to use imports in this form, otherwise isinstance checks fail on helper classes
from helpers.truss_patch.model_container_patch_applier import (  # noqa
    ModelContainerPatchApplier,
)
from helpers.custom_types import (  # noqa
    Action,
    ConfigPatch,
    EnvVarPatch,
    ExternalDataPatch,
    ModelCodePatch,
    PackagePatch,
    Patch,
    PatchType,
)


@pytest.fixture
def patch_applier(truss_container_fs):
    return ModelContainerPatchApplier(truss_container_fs / "app", mock.Mock())


def test_patch_applier_model_code_patch_add(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(action=Action.ADD, path="dummy", content=""),
    )
    patch_applier(patch, os.environ.copy())
    assert (truss_container_fs / "app" / "model" / "dummy").exists()


def test_patch_applier_model_code_patch_remove(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(action=Action.REMOVE, path="model.py"),
    )
    assert (truss_container_fs / "app" / "model" / "model.py").exists()
    patch_applier(patch, os.environ.copy())
    assert not (truss_container_fs / "app" / "model" / "model.py").exists()


def test_patch_applier_model_code_patch_update(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    new_model_file_content = """
    class Model:
        pass
    """
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE, path="model.py", content=new_model_file_content
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (
        truss_container_fs / "app" / "model" / "model.py"
    ).read_text() == new_model_file_content


def test_patch_applier_package_patch_add(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    patch = Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(
            action=Action.ADD, path="test_package/test.py", content="foobar"
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (truss_container_fs / "packages" / "test_package" / "test.py").exists()


def test_patch_applier_package_patch_remove(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    patch = Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(action=Action.REMOVE, path="test_package/test.py"),
    )
    assert (truss_container_fs / "packages" / "test_package" / "test.py").exists()
    patch_applier(patch, os.environ.copy())
    assert not (truss_container_fs / "packages" / "test_package" / "test.py").exists()


def test_patch_applier_package_patch_update(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    new_package_content = """X = 2"""
    patch = Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(
            action=Action.UPDATE,
            path="test_package/test.py",
            content=new_package_content,
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (
        truss_container_fs / "packages" / "test_package" / "test.py"
    ).read_text() == new_package_content


def test_patch_applier_config_patch_update(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    new_config_dict = {"model_name": "foobar"}
    patch = Patch(
        type=PatchType.CONFIG,
        body=ConfigPatch(action=Action.UPDATE, config=new_config_dict),
    )
    patch_applier(patch, os.environ.copy())
    new_config = TrussConfig.from_yaml(truss_container_fs / "app" / "config.yaml")
    assert new_config.model_name == "foobar"


def test_patch_applier_env_var_patch_update(patch_applier: ModelContainerPatchApplier):
    env_var_dict = {"FOO": "BAR"}
    patch = Patch(
        type=PatchType.ENVIRONMENT_VARIABLE,
        body=EnvVarPatch(action=Action.UPDATE, item={"FOO": "BAR-PATCHED"}),
    )
    patch_applier(patch, env_var_dict)
    assert env_var_dict["FOO"] == "BAR-PATCHED"


def test_patch_applier_env_var_patch_add(patch_applier: ModelContainerPatchApplier):
    env_var_dict = {"FOO": "BAR"}
    patch = Patch(
        type=PatchType.ENVIRONMENT_VARIABLE,
        body=EnvVarPatch(action=Action.ADD, item={"BAR": "FOO"}),
    )
    patch_applier(patch, env_var_dict)
    assert env_var_dict["FOO"] == "BAR"
    assert env_var_dict["BAR"] == "FOO"


def test_patch_applier_env_var_patch_remove(patch_applier: ModelContainerPatchApplier):
    env_var_dict = {"FOO": "BAR"}
    patch = Patch(
        type=PatchType.ENVIRONMENT_VARIABLE,
        body=EnvVarPatch(action=Action.REMOVE, item={"FOO": "BAR"}),
    )
    patch_applier(patch, env_var_dict)
    with pytest.raises(KeyError):
        _ = env_var_dict["FOO"]


@pytest.mark.parametrize(
    "patch_type, body_cls",
    [(PatchType.MODEL_CODE, ModelCodePatch), (PatchType.PACKAGE, PackagePatch)],
)
@pytest.mark.parametrize("action", [Action.ADD, Action.UPDATE])
def test_patch_applier_rejects_path_traversal_write(
    patch_applier: ModelContainerPatchApplier,
    truss_container_fs,
    patch_type,
    body_cls,
    action,
):
    # A patch path escaping the target directory (e.g. via `..`) must be
    # rejected instead of writing outside it (issue #2532).
    outside_target = truss_container_fs / "escaped.py"
    patch = Patch(
        type=patch_type,
        body=body_cls(action=action, path="../../escaped.py", content="pwned"),
    )
    with pytest.raises(ValueError):
        patch_applier(patch, os.environ.copy())
    assert not outside_target.exists()


def test_patch_applier_rejects_absolute_path_write(
    patch_applier: ModelContainerPatchApplier, truss_container_fs, tmp_path
):
    # Joining an absolute path onto the target dir collapses to the absolute
    # path, another way to escape; it must be rejected too.
    outside_target = tmp_path / "abs_escaped.py"
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.ADD, path=str(outside_target), content="pwned"
        ),
    )
    with pytest.raises(ValueError):
        patch_applier(patch, os.environ.copy())
    assert not outside_target.exists()


def test_patch_applier_rejects_path_traversal_remove(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    # A REMOVE patch must not be able to delete files outside the target dir.
    victim = truss_container_fs / "app" / "config.yaml"
    assert victim.exists()
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(action=Action.REMOVE, path="../config.yaml"),
    )
    with pytest.raises(ValueError):
        patch_applier(patch, os.environ.copy())
    assert victim.exists()


def test_patch_applier_allows_nested_subdirectory_path(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    # Legitimate nested paths within the target dir must still work.
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(action=Action.ADD, path="nested/dir/new.py", content="ok"),
    )
    patch_applier(patch, os.environ.copy())
    assert (
        truss_container_fs / "app" / "model" / "nested" / "dir" / "new.py"
    ).read_text() == "ok"


def test_patch_applier_external_data_patch_add(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    patch = Patch(
        type=PatchType.EXTERNAL_DATA,
        body=ExternalDataPatch(
            action=Action.ADD,
            item={
                "url": "https://raw.githubusercontent.com/basetenlabs/truss/main/docs/favicon.svg",
                "local_data_path": "truss_icon",
            },
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (truss_container_fs / "app" / "data" / "truss_icon").exists()
