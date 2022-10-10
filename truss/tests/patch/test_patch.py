from pathlib import Path

from truss.patch.calc_patch import calc_truss_patch
from truss.patch.signature import calc_truss_signature
from truss.templates.control.control.helpers.types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
)


def test_calc_truss_patch_unsupported(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    (custom_model_truss_dir / "dummy").touch()
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)
    assert patches is None


def test_calc_truss_patch_add_file(custom_model_truss_dir: Path):
    prev_sign = calc_truss_signature(custom_model_truss_dir)
    (custom_model_truss_dir / "model" / "dummy").touch()
    patches = calc_truss_patch(custom_model_truss_dir, prev_sign)

    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="dummy",
            content="",
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
            action=Action.UPDATE,
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
