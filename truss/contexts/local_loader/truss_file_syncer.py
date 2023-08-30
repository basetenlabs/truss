import logging
from pathlib import Path
from threading import Thread

import rich


class TrussFilesSyncer(Thread):
    """Daemon thread that watches for changes in the user's Truss and syncs to running service."""

    def __init__(
        self,
        watch_path: Path,
        remote,
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

        # disable watchfiles logger
        logging.getLogger("watchfiles.main").disabled = True

        rich.print(f"🚰 Attempting to sync truss at '{self.watch_path}' with remote")
        self.remote.patch(self.watch_path, self._logger)

        rich.print(f"👀 Watching for changes to truss at '{self.watch_path}' ...")
        for _ in watch(
            self.watch_path, watch_filter=self.watch_filter, raise_interrupt=False
        ):
            self.remote.patch(self.watch_path, self._logger)
