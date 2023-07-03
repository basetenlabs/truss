import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from truss.truss_config import TrussConfig

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
from helpers.types import (  # noqa
    Action,
    ConfigPatch,
    EnvVarPatch,
    ExternalDataPatch,
    ModelCodePatch,
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
        body=ModelCodePatch(
            action=Action.ADD,
            path="dummy",
            content="",
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (truss_container_fs / "app" / "model" / "dummy").exists()


def test_patch_applier_model_code_patch_remove(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.REMOVE,
            path="model.py",
        ),
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
            action=Action.UPDATE,
            path="model.py",
            content=new_model_file_content,
        ),
    )
    patch_applier(patch, os.environ.copy())
    with (truss_container_fs / "app" / "model" / "model.py").open() as model_file:
        assert model_file.read() == new_model_file_content


def test_patch_applier_config_patch_update(
    patch_applier: ModelContainerPatchApplier, truss_container_fs
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
    new_config = TrussConfig.from_yaml(truss_container_fs / "app" / "config.yaml")
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
    truss_container_fs,
):
    patch = Patch(
        type=PatchType.EXTERNAL_DATA,
        body=ExternalDataPatch(
            action=Action.ADD,
            item={
                "url": "https://raw.githubusercontent.com/basetenlabs/truss/main/docs/assets/truss_logo_horizontal.png",
                "local_data_path": "truss_icon",
            },
        ),
    )
    patch_applier(patch, os.environ.copy())
    assert (truss_container_fs / "app" / "data" / "truss_icon").exists()
