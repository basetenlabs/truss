import logging
import subprocess
from pathlib import Path
from typing import Optional

from helpers.custom_types import (
    Action,
    ConfigPatch,
    EnvVarPatch,
    ExternalDataPatch,
    ModelCodePatch,
    PackagePatch,
    Patch,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from helpers.errors import UnsupportedPatch
from helpers.truss_patch.model_code_patch_applier import apply_code_patch

from truss.base.truss_config import ExternalData, ExternalDataItem, TrussConfig
from truss.util.download import download_external_data


class ModelContainerPatchApplier:
    """Applies patches to container running a truss.
    This should be compatible with TrussDirPatchApplier.
    """

    def __init__(
        self,
        inference_server_home: Path,
        app_logger: logging.Logger,
        pip_path: Optional[str] = None,  # Only meant for testing
    ) -> None:
        self._inference_server_home = inference_server_home
        self._model_module_dir = (
            self._inference_server_home / self._truss_config.model_module_dir
        )
        self._bundled_packages_dir = (
            self._inference_server_home / ".." / self._truss_config.bundled_packages_dir
        ).resolve()
        self._data_dir = self._inference_server_home / self._truss_config.data_dir
        self._app_logger = app_logger
        self._pip_path_cached = None
        if pip_path is not None:
            self._pip_path_cached = "pip"

    def __call__(self, patch: Patch, inf_env: dict):
        self._app_logger.debug(f"Applying patch {patch.to_dict()}")
        if isinstance(patch.body, ModelCodePatch):
            model_code_patch: ModelCodePatch = patch.body
            apply_code_patch(self._model_module_dir, model_code_patch, self._app_logger)
        elif isinstance(patch.body, PythonRequirementPatch):
            py_req_patch: PythonRequirementPatch = patch.body
            self._apply_python_requirement_patch(py_req_patch)
        elif isinstance(patch.body, SystemPackagePatch):
            sys_pkg_patch: SystemPackagePatch = patch.body
            self._apply_system_package_patch(sys_pkg_patch)
        elif isinstance(patch.body, ConfigPatch):
            config_patch: ConfigPatch = patch.body
            self._apply_config_patch(config_patch)
        elif isinstance(patch.body, EnvVarPatch):
            env_var_patch: EnvVarPatch = patch.body
            self._apply_env_var_patch(env_var_patch, inf_env)
        elif isinstance(patch.body, ExternalDataPatch):
            external_data_patch: ExternalDataPatch = patch.body
            self._apply_external_data_patch(external_data_patch)
        elif isinstance(patch.body, PackagePatch):
            package_patch: PackagePatch = patch.body
            apply_code_patch(
                self._bundled_packages_dir, package_patch, self._app_logger
            )
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
                    "--upgrade",
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
                ["apt", "remove", "-y", system_package_patch.package], check=True
            )
        elif action in [Action.ADD, Action.UPDATE]:
            subprocess.run(["apt", "update"], check=True)
            subprocess.run(
                ["apt", "install", "-y", system_package_patch.package], check=True
            )
        else:
            raise ValueError(f"Unknown python requirement patch action {action}")

    def _apply_config_patch(self, config_patch: ConfigPatch):
        self._app_logger.debug(f"Applying config patch {config_patch.to_dict()}")
        TrussConfig.from_dict(config_patch.config).write_to_yaml_file(
            Path(self._inference_server_home / config_patch.path)
        )

    def _apply_env_var_patch(self, env_var_patch: EnvVarPatch, inf_env: dict):
        self._app_logger.debug(
            f"Applying environment variable patch {env_var_patch.to_dict()}"
        )
        action = env_var_patch.action
        ((env_var_name, env_var_value),) = env_var_patch.item.items()

        if action == Action.REMOVE:
            inf_env.pop(env_var_name, None)
        elif action in [Action.ADD, Action.UPDATE]:
            inf_env.update({env_var_name: env_var_value})
        else:
            raise ValueError(f"Unknown patch action {action}")

    def _apply_external_data_patch(self, external_data_patch: ExternalDataPatch):
        self._app_logger.debug(f"Applying external data patch {external_data_patch}")
        action = external_data_patch.action
        try:
            item = ExternalDataItem.model_validate(external_data_patch.item)
        except Exception:
            item = ExternalDataItem.from_dict(external_data_patch.item)  # type: ignore[attr-defined]

        if action == Action.REMOVE:
            filepath = self._data_dir / item.local_data_path
            if not filepath.exists():
                self._app_logger.warning(
                    f"Could not delete file {filepath}: not found."
                )
            else:
                self._app_logger.debug(f"Deleting file {filepath}")
                filepath.unlink()
        elif action == Action.ADD:
            download_external_data(ExternalData([item]), self._data_dir)
        else:
            raise ValueError(f"Unknown patch action {action}")


def _identify_pip_path() -> str:
    if Path("/usr/local/bin/pip3").exists():
        return "/usr/local/bin/pip3"

    if Path("/usr/local/bin/pip").exists():
        return "/usr/local/bin/pip"

    raise RuntimeError("Unable to find pip, make sure it's installed.")
