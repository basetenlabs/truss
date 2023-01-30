import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

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
from helpers.patch_applier import PatchApplier  # noqa
from helpers.types import Action, ModelCodePatch, Patch, PatchType  # noqa


@pytest.fixture
def patch_applier(truss_container_fs):
    return PatchApplier(truss_container_fs / "app", Mock())


def test_patch_applier_add(patch_applier: PatchApplier, truss_container_fs):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.ADD,
            path="dummy",
            content="",
        ),
    )
    patch_applier.apply_patch(patch)
    assert (truss_container_fs / "app" / "model" / "dummy").exists()


def test_patch_applier_remove(patch_applier: PatchApplier, truss_container_fs):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.REMOVE,
            path="model.py",
        ),
    )
    assert (truss_container_fs / "app" / "model" / "model.py").exists()
    patch_applier.apply_patch(patch)
    assert not (truss_container_fs / "app" / "model" / "model.py").exists()


def test_patch_applier_update(patch_applier: PatchApplier, truss_container_fs):
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
    patch_applier.apply_patch(patch)
    with (truss_container_fs / "app" / "model" / "model.py").open() as model_file:
        assert model_file.read() == new_model_file_content
