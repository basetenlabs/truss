import logging
from pathlib import Path
from threading import Thread
from typing import Optional

from truss.remote.baseten import BasetenRemote


class TrussFilesSyncer(Thread):
    """Daemon thread that watches for changes in the user's Truss and syncs to running service."""

    def __init__(
        self,
        watch_path: Path,
        remote: Optional[BasetenRemote],
    ) -> None:
        from truss.util.path import is_ignored, load_trussignore_patterns

        super().__init__(daemon=True)
        self._logger = logging.Logger(__name__)
        self.watch_path = watch_path
        self.remote = remote
        self.watch_filter = lambda _, path: not is_ignored(
            Path(path),
            load_trussignore_patterns(),
        )

    def run(self) -> None:
        """Watch for files in background and apply appropriate patches."""
        from watchfiles import watch

        if not self.remote:
            return

        for changes in watch(
            self.watch_path, watch_filter=self.watch_filter, raise_interrupt=False
        ):
            # print(changes)
            self.remote.watch(self.watch_path, self._logger)
