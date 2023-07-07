import tarfile
import tempfile
from pathlib import Path
from typing import IO, Any, Callable

from rich.progress import Progress


class ReadProgressIndicatorFileHandle:
    def __init__(
        self, file: IO[bytes], progress_callback: Callable[[int], Any]
    ) -> None:
        self._file = file
        self._progress_callback = progress_callback

    def read(self, size=-1):
        data = self._file.read(size)
        self._progress_callback(len(data))
        return data

    def __getattr__(self, attr):
        return getattr(self._file, attr)


def create_tar_with_progress_bar(source_dir: Path, delete=True):
    total_size = sum(f.stat().st_size for f in source_dir.glob("**/*") if f.is_file())

    # Keeping the .tgz suffix for backwards compatibility even though
    # this tar is uncompressed for upload
    temp_file = tempfile.NamedTemporaryFile(suffix=".tgz", delete=delete)
    with tarfile.open(temp_file.name, "w:") as tar:

        # Create a new progress bar
        progress = Progress()

        # Add a new task to the progress bar
        task_id = progress.add_task("[cyan]Compressing...", total=total_size)

        with progress:

            def file_read_progress_callback(bytes_read: int):
                # Update the progress bar
                progress.update(task_id, advance=bytes_read)

            for file_path in source_dir.glob("**/*"):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(source_dir))
                    with file_path.open("rb") as file_obj:
                        file_obj_with_progress = ReadProgressIndicatorFileHandle(
                            file_obj, file_read_progress_callback
                        )
                        tarinfo = tar.gettarinfo(name=str(file_path), arcname=arcname)
                        tar.addfile(
                            tarinfo=tarinfo, fileobj=file_obj_with_progress  # type: ignore[arg-type]
                        )
    return temp_file
