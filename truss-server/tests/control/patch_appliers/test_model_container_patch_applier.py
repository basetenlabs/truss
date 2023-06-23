from unittest.mock import Mock

import pytest
from truss.core.patch.types import Action, ModelCodePatch, Patch, PatchType  # noqa
from truss.server.control.patch_appliers.model_container_patch_applier import (  # noqa
    ModelContainerPatchApplier,
)


@pytest.fixture
def patch_applier(truss_context_dir):
    return ModelContainerPatchApplier(truss_context_dir, Mock())


def test_patch_applier_add(
    patch_applier: ModelContainerPatchApplier, truss_context_dir
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
    assert (truss_context_dir / "model" / "dummy").exists()


def test_patch_applier_remove(
    patch_applier: ModelContainerPatchApplier, truss_context_dir
):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.REMOVE,
            path="model.py",
        ),
    )
    assert (truss_context_dir / "model" / "model.py").exists()
    patch_applier(patch)
    assert not (truss_context_dir / "model" / "model.py").exists()


def test_patch_applier_update(
    patch_applier: ModelContainerPatchApplier, truss_context_dir
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
    with (truss_context_dir / "model" / "model.py").open() as model_file:
        assert model_file.read() == new_model_file_content
