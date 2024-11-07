import logging
import subprocess
from pathlib import Path
from typing import List

from truss.base.truss_config import TrussConfig
from truss.templates.control.control.helpers.custom_types import (
    Action,
    ModelCodePatch,
    Patch,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.templates.control.control.helpers.errors import UnsupportedPatch
from truss.templates.control.control.helpers.truss_patch.model_code_patch_applier import (
    apply_code_patch,
)


class LocalTrussPatchApplier:
    """Applies patches to a truss server running locally.
    This should be compatible with ModelContainerPatchApplier.
    """

    def __init__(self, truss_dir: Path, env_exe, logger: logging.Logger) -> None:
        self._truss_dir = truss_dir
        self._env_exe = env_exe
        self._logger = logger

    def __call__(self, patches: List[Patch]):
        for patch in patches:
            self._logger.debug(f"Applying patch {patch.to_dict()}")
            if isinstance(patch.body, ModelCodePatch):
                model_code_patch: ModelCodePatch = patch.body
                model_module_dir = self._truss_dir / self._truss_config.model_module_dir
                apply_code_patch(model_module_dir, model_code_patch, self._logger)
            elif isinstance(patch.body, PythonRequirementPatch):
                py_req_patch: PythonRequirementPatch = patch.body
                self._apply_python_requirement_patch(py_req_patch)
            elif isinstance(patch.body, SystemPackagePatch):
                self._logger.info(
                    "System package patches are not supported for local server"
                )
            else:
                raise UnsupportedPatch(f"Unknown patch type {patch.type}")

    @property
    def _truss_config(self) -> TrussConfig:
        return TrussConfig.from_yaml(self._truss_dir / "config.yaml")

    def _apply_python_requirement_patch(
        self, python_requirement_patch: PythonRequirementPatch
    ):
        self._logger.debug(
            f"Applying python requirement patch {python_requirement_patch.to_dict()}"
        )
        action = python_requirement_patch.action

        if action == Action.REMOVE:
            subprocess.run(
                [
                    self._env_exe,
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    python_requirement_patch.requirement,
                ],
                check=True,
            )
        elif action in [Action.ADD, Action.UPDATE]:
            subprocess.run(
                [
                    self._env_exe,
                    "-m",
                    "pip",
                    "install",
                    python_requirement_patch.requirement,
                ],
                check=True,
            )
        else:
            raise ValueError(f"Unknown python requirement patch action {action}")
