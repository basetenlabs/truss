from unittest.mock import Mock

import pytest
from truss_common.patch.types import Action, ModelCodePatch, Patch, PatchType

# Have to use imports in this form, otherwise isinstance checks fail on helper classes
from truss_server.control.helpers.truss_patch.model_container_patch_applier import (  # noqa
    ModelContainerPatchApplier,
)


@pytest.fixture
def patch_applier(truss_container_fs):
    return ModelContainerPatchApplier(truss_container_fs / "app", Mock())


def test_patch_applier_add(
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
    patch_applier(patch)
    assert (truss_container_fs / "app" / "model" / "dummy").exists()


def test_patch_applier_remove(
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
    patch_applier(patch)
    assert not (truss_container_fs / "app" / "model" / "model.py").exists()


def test_patch_applier_update(
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
    patch_applier(patch)
    with (truss_container_fs / "app" / "model" / "model.py").open() as model_file:
        assert model_file.read() == new_model_file_content
