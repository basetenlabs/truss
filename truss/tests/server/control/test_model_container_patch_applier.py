import os
from unittest import mock

import pytest
from truss.server.control.patch.model_container_patch_applier import (
    ModelContainerPatchApplier,
)
from truss.server.control.patch.types import (
    Action,
    ConfigPatch,
    EnvVarPatch,
    ExternalDataPatch,
    ModelCodePatch,
    PackagePatch,
    Patch,
    PatchType,
)
from truss.truss_config import TrussConfig


@pytest.fixture
def patch_applier(tmp_truss_dir):
    return ModelContainerPatchApplier(tmp_truss_dir, mock.Mock())


def test_patch_applier_model_code_patch_add(
    patch_applier: ModelContainerPatchApplier, tmp_truss_dir
):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.ADD,
            path="dummy",
            content="",
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (tmp_truss_dir / "model" / "dummy").exists()


def test_patch_applier_model_code_patch_remove(
    patch_applier: ModelContainerPatchApplier, tmp_truss_dir
):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.REMOVE,
            path="model.py",
        ),
    )
    assert (tmp_truss_dir / "model" / "model.py").exists()
    patch_applier(patch, os.environ.copy())
    assert not (tmp_truss_dir / "model" / "model.py").exists()


def test_patch_applier_model_code_patch_update(
    patch_applier: ModelContainerPatchApplier, tmp_truss_dir
):
    new_model_file_content = """
    class Model:
        pass
    """
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="model.py",
            content=new_model_file_content,
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (tmp_truss_dir / "model" / "model.py").read_text() == new_model_file_content


def test_patch_applier_package_patch_add(
    patch_applier: ModelContainerPatchApplier, tmp_truss_dir
):
    patch = Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(
            action=Action.ADD,
            path="test_package/test.py",
            content="foobar",
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (tmp_truss_dir / "packages" / "test_package" / "test.py").exists()


def test_patch_applier_package_patch_remove(
    patch_applier: ModelContainerPatchApplier,
    tmp_truss_dir,
):
    patch = Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(
            action=Action.REMOVE,
            path="test_package/test.py",
        ),
    )
    assert (tmp_truss_dir / "packages" / "test_package" / "test.py").exists()
    patch_applier(patch, os.environ.copy())
    assert not (tmp_truss_dir / "packages" / "test_package" / "test.py").exists()


def test_patch_applier_package_patch_update(
    patch_applier: ModelContainerPatchApplier,
    tmp_truss_dir,
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
        tmp_truss_dir / "packages" / "test_package" / "test.py"
    ).read_text() == new_package_content


def test_patch_applier_config_patch_update(
    patch_applier: ModelContainerPatchApplier, tmp_truss_dir
):
    new_config_dict = {"model_name": "foobar"}
    patch = Patch(
        type=PatchType.CONFIG,
        body=ConfigPatch(
            action=Action.UPDATE,
            config=new_config_dict,
        ),
    )
    patch_applier(patch, os.environ.copy())
    new_config = TrussConfig.from_yaml(tmp_truss_dir / "config.yaml")
    assert new_config.model_name == "foobar"


def test_patch_applier_env_var_patch_update(
    patch_applier: ModelContainerPatchApplier,
):
    env_var_dict = {"FOO": "BAR"}
    patch = Patch(
        type=PatchType.ENVIRONMENT_VARIABLE,
        body=EnvVarPatch(
            action=Action.UPDATE,
            item={"FOO": "BAR-PATCHED"},
        ),
    )
    patch_applier(patch, env_var_dict)
    assert env_var_dict["FOO"] == "BAR-PATCHED"


def test_patch_applier_env_var_patch_add(
    patch_applier: ModelContainerPatchApplier,
):
    env_var_dict = {"FOO": "BAR"}
    patch = Patch(
        type=PatchType.ENVIRONMENT_VARIABLE,
        body=EnvVarPatch(
            action=Action.ADD,
            item={"BAR": "FOO"},
        ),
    )
    patch_applier(patch, env_var_dict)
    assert env_var_dict["FOO"] == "BAR"
    assert env_var_dict["BAR"] == "FOO"


def test_patch_applier_env_var_patch_remove(
    patch_applier: ModelContainerPatchApplier,
):
    env_var_dict = {"FOO": "BAR"}
    patch = Patch(
        type=PatchType.ENVIRONMENT_VARIABLE,
        body=EnvVarPatch(
            action=Action.REMOVE,
            item={"FOO": "BAR"},
        ),
    )
    patch_applier(patch, env_var_dict)
    with pytest.raises(KeyError):
        _ = env_var_dict["FOO"]


def test_patch_applier_external_data_patch_add(
    patch_applier: ModelContainerPatchApplier,
    tmp_truss_dir,
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
    assert (tmp_truss_dir / "data" / "truss_icon").exists()
