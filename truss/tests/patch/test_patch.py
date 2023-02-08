from pathlib import Path
from typing import Any, Callable, List, Optional

from truss.patch.calc_patch import calc_truss_patch
from truss.patch.signature import calc_truss_signature
from truss.templates.control.control.helpers.types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.truss_config import TrussConfig


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
            action=Action.ADD,
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


def test_calc_config_patches_add_python_requirement(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        lambda config: config.requirements.append("requests==1.0.0"),
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(
            action=Action.ADD,
            requirement="requests==1.0.0",
        ),
    )


def test_calc_config_patches_remove_python_requirement(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=lambda config: config.requirements.append("requests==1.0.0"),
        config_op=lambda config: config.requirements.clear(),
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(
            action=Action.REMOVE,
            requirement="requests",
        ),
    )


def test_calc_config_patches_update_python_requirement(custom_model_truss_dir: Path):
    def update_requests_version(config: TrussConfig):
        config.requirements[0] = "requests==2.0.0"

    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=lambda config: config.requirements.append("requests==1.0.0"),
        config_op=update_requests_version,
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(
            action=Action.UPDATE,
            requirement="requests==2.0.0",
        ),
    )


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
    assert len(patches) == 3
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


def test_calc_config_patches_non_python_or_system_requirement_change(
    custom_model_truss_dir: Path,
):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_op=lambda config: config.environment_variables.update({"foo": "bar"}),
    )
    assert patches is None


def test_calc_config_patches_add_system_package(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        lambda config: config.system_packages.append("curl"),
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.SYSTEM_PACKAGE,
        body=SystemPackagePatch(
            action=Action.ADD,
            package="curl",
        ),
    )


def test_calc_config_patches_remove_system_package(custom_model_truss_dir: Path):
    patches = _apply_config_change_and_calc_patches(
        custom_model_truss_dir,
        config_pre_op=lambda config: config.system_packages.append("curl"),
        config_op=lambda config: config.system_packages.clear(),
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch == Patch(
        type=PatchType.SYSTEM_PACKAGE,
        body=SystemPackagePatch(
            action=Action.REMOVE,
            package="curl",
        ),
    )


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
