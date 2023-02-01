import subprocess
from pathlib import Path

from helpers.errors import UnsupportedPatch
from helpers.types import (
    Action,
    ModelCodePatch,
    Patch,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.truss_config import TrussConfig


class PatchApplier:
    def __init__(
        self,
        inference_server_home: Path,
        app_logger,
        pip_path: str = None,  # Only meant for testing
    ) -> None:
        self._inference_server_home = inference_server_home
        self._model_module_dir = (
            self._inference_server_home / self._truss_config.model_module_dir
        )
        self._app_logger = app_logger
        self._pip_path_cached = None
        if pip_path is not None:
            self._pip_path_cached = "pip"

    def apply_patch(self, patch: Patch):
        self._app_logger.debug(f"Applying patch {patch.to_dict()}")
        if isinstance(patch.body, ModelCodePatch):
            model_code_patch: ModelCodePatch = patch.body
            self._apply_model_code_patch(model_code_patch)
        elif isinstance(patch.body, PythonRequirementPatch):
            py_req_patch: PythonRequirementPatch = patch.body
            self._apply_python_requirement_patch(py_req_patch)
        elif isinstance(patch.body, SystemPackagePatch):
            sys_pkg_patch: SystemPackagePatch = patch.body
            self._apply_system_package_patch(sys_pkg_patch)
        else:
            raise UnsupportedPatch(f"Unknown patch type {patch.type}")

    @property
    def _truss_config(self) -> TrussConfig:
        return TrussConfig.from_yaml(self._inference_server_home / "config.yaml")

    @property
    def _pip_path(self) -> str:
        if self._pip_path_cached is None:
            self._pip_path_cached = _identify_pip_path()
        return self._pip_path_cached

    def _apply_model_code_patch(self, model_code_patch: ModelCodePatch):
        self._app_logger.debug(
            f"Applying model code patch {model_code_patch.to_dict()}"
        )
        action = model_code_patch.action
        filepath: Path = self._model_module_dir / model_code_patch.path
        if action in [Action.ADD, Action.UPDATE]:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self._app_logger.info(f"Updating file {filepath}")
            with filepath.open("w") as file:
                content = model_code_patch.content
                if content is None:
                    raise ValueError(
                        "Invalid patch: content of a model code update patch should not be None."
                    )
                file.write(content)

        elif action == Action.REMOVE:
            if not filepath.exists():
                self._app_logger.warning(
                    f"Could not delete file {filepath}: not found."
                )
            else:
                self._app_logger.info(f"Deleting file {filepath}")
                filepath.unlink()
        else:
            raise ValueError(f"Unknown model code patch action {action}")

    def _apply_python_requirement_patch(
        self, python_requirement_patch: PythonRequirementPatch
    ):
        self._app_logger.debug(
            f"Applying python requirement patch {python_requirement_patch.to_dict()}"
        )
        action = python_requirement_patch.action

        if action == Action.REMOVE:
            subprocess.run(
                [
                    self._pip_path,
                    "uninstall",
                    "-y",
                    python_requirement_patch.requirement,
                ],
                check=True,
            )
        elif action in [Action.ADD, Action.UPDATE]:
            subprocess.run(
                [
                    self._pip_path,
                    "install",
                    python_requirement_patch.requirement,
                ],
                check=True,
            )
        else:
            raise ValueError(f"Unknown python requirement patch action {action}")

    def _apply_system_package_patch(self, system_package_patch: SystemPackagePatch):
        self._app_logger.debug(
            f"Applying system package patch {system_package_patch.to_dict()}"
        )
        action = system_package_patch.action

        if action == Action.REMOVE:
            subprocess.run(
                [
                    "apt",
                    "remove",
                    "-y",
                    system_package_patch.package,
                ],
                check=True,
            )
        elif action in [Action.ADD, Action.UPDATE]:
            subprocess.run(
                [
                    "apt",
                    "update",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "apt",
                    "install",
                    "-y",
                    system_package_patch.package,
                ],
                check=True,
            )
        else:
            raise ValueError(f"Unknown python requirement patch action {action}")


def _identify_pip_path() -> str:
    if Path("/usr/local/bin/pip3").exists():
        return "/usr/local/bin/pip3"

    if Path("/usr/local/bin/pip").exists():
        return "/usr/local/bin/pip"

    raise RuntimeError("Unable to find pip, make sure it's installed.")
