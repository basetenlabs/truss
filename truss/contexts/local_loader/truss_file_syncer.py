import logging
from pathlib import Path
from threading import Thread
from typing import List, Optional

from truss.constants import CONFIG_FILE
from truss.patch.calc_patch import calc_config_patches
from truss.patch.local_truss_patch_applier import LocalTrussPatchApplier
from truss.templates.control.control.helpers.types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
)
from truss.truss_config import TrussConfig
from truss.truss_spec import TrussSpec
from truss.util.path import file_content, is_ignored, load_trussignore_patterns
from watchfiles import Change, watch

OP_2_ACTION = {
    Change.added: Action.ADD,
    Change.deleted: Action.REMOVE,
    Change.modified: Action.UPDATE,
}


class TrussPatchEmitter:
    def __init__(
        self,
        truss_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self._truss_dir = truss_dir
        self._config = TrussConfig.from_yaml(self._truss_dir / CONFIG_FILE)
        self._logger = logger

    def __call__(self, op, path: Path) -> Optional[List[Patch]]:
        truss_spec = TrussSpec(self._truss_dir)
        model_module_path = str(
            truss_spec.model_module_dir.relative_to(self._truss_dir)
        )
        if str(path).startswith(model_module_path):
            rel_path = str(path.relative_to(model_module_path))
            return [
                Patch(
                    type=PatchType.MODEL_CODE,
                    body=ModelCodePatch(
                        OP_2_ACTION[op],
                        rel_path,
                        file_content(self._truss_dir / path),
                    ),
                )
            ]
        if str(path) == CONFIG_FILE:
            new_config = TrussConfig.from_yaml(self._truss_dir / CONFIG_FILE)
            config_patches = calc_config_patches(self._config, new_config)
            self._config = new_config
            return config_patches or None
        return None


class TrussFilesSyncer(Thread):
    """Daemon thread that watches for changes in the user's Truss and syncs to running service."""

    def __init__(self, watch_path: Path, patch_applier: LocalTrussPatchApplier) -> None:
        super().__init__(daemon=True)
        self._logger = logging.Logger(__name__)
        self.watch_path = watch_path
        self.patch_emitter = TrussPatchEmitter(self.watch_path, self._logger)
        self.patch_applier = patch_applier
        self.watch_filter = lambda _, path: not is_ignored(
            Path(path),
            load_trussignore_patterns(),
        )

    def run(self) -> None:
        """Watch for files in background and apply appropriate patches."""
        for changes in watch(self.watch_path, watch_filter=self.watch_filter):
            for change in changes:
                op, path = change
                rel_path = Path(path).relative_to(self.watch_path.resolve())
                patches = self.patch_emitter(op, rel_path)
                if patches:
                    self.patch_applier(patches)
