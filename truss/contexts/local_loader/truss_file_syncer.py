from pathlib import Path
from threading import Thread

from truss.util.path import copy_file_path
from watchfiles import Change, watch


class TrussFilesSyncer(Thread):
    def __init__(self, watch_path: Path, mirror_path: Path) -> None:
        super().__init__(daemon=True)
        self.watch_path = watch_path
        self.mirror_path = mirror_path

    def run(self):
        for changes in watch(str(self.watch_path)):
            for change in changes:
                op, path = change
                rel_path = Path(path).relative_to(self.watch_path.resolve())
                if op == Change.modified:
                    copy_file_path(
                        self.watch_path / rel_path,
                        self.mirror_path / rel_path,
                    )
