import logging
from pathlib import Path
from threading import Thread
from typing import Optional

from truss.patch.truss_dir_patch_applier import TrussDirPatchApplier
from truss.templates.control.control.helpers.types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
)
from truss.truss_spec import TrussSpec
from watchfiles import Change, watch

OP_2_ACTION = {
    Change.added: Action.ADD,
    Change.deleted: Action.REMOVE,
    Change.modified: Action.UPDATE,
}


def _relative_to(path: str, relative_to_path: str):
    return str(Path(path).relative_to(relative_to_path))


def _file_content(path: Path) -> str:
    with path.open() as file:
        return file.read()


class TrussPatchEmitter:
    def __init__(
        self,
        truss_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.truss_dir = truss_dir
        self._logger = logger

    def emit(self, op, path: Path) -> Optional[Patch]:
        truss_spec = TrussSpec(self.truss_dir)
        model_module_path = str(truss_spec.model_module_dir.relative_to(self.truss_dir))
        print(model_module_path)
        print(path)
        if str(path).startswith(model_module_path):
            return Patch(
                type=PatchType.MODEL_CODE,
                body=ModelCodePatch(
                    OP_2_ACTION[op], str(path), _file_content(self.truss_dir / path)
                ),
            )
        return None


class TrussFilesSyncer(Thread):
    """Daemon thread that watches for changes in the user's Truss and syncs to running service."""

    def __init__(self, watch_path: Path, patch_applier: TrussDirPatchApplier) -> None:
        super().__init__(daemon=True)
        self._logger = logging.Logger(__name__)
        self.watch_path = watch_path
        self.patch_emitter = TrussPatchEmitter(self.watch_path, self._logger)
        self.patch_applier = patch_applier

    def run(self) -> None:
        """Watch for files in background and apply appropriate patches."""
        for changes in watch(str(self.watch_path)):
            for change in changes:
                op, path = change
                if not str(path).endswith(".tmp"):
                    rel_path = Path(path).relative_to(self.watch_path.resolve())
                    patch = self.patch_emitter.emit(op, rel_path)
                    if patch:
                        logging.Logger(__name__).info(patch)
                        self.patch_applier([patch])
