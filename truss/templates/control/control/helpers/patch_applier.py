from pathlib import Path

from helpers.types import Action, ModelCodePatch, Patch
from truss.truss_config import TrussConfig


class PatchApplier:
    def __init__(
        self,
        inference_server_home: Path,
    ) -> None:
        self._inference_server_home = inference_server_home
        self._model_module_dir = (
            self._inference_server_home / self._truss_config.model_module_dir
        )

    def apply_patch(self, patch: Patch):
        if isinstance(patch.body, ModelCodePatch):
            model_code_patch: ModelCodePatch = patch.body
            self._apply_model_code_patch(model_code_patch)
        else:
            raise ValueError(f"Unknown patch type {patch.type}")

    @property
    def _truss_config(self) -> TrussConfig:
        return TrussConfig.from_yaml(self._inference_server_home / "config.yaml")

    def _apply_model_code_patch(self, model_code_patch: ModelCodePatch):
        action = model_code_patch.action
        filepath: Path = self._model_module_dir / model_code_patch.path
        if action == Action.UPDATE:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open("w") as file:
                content = model_code_patch.content
                if content is None:
                    content = ""
                file.write(content)

        elif action == Action.REMOVE:
            # todo: log error when file is missing
            filepath.unlink(missing_ok=True)
        else:
            raise ValueError(f"Unknown model code patch action {action}")
