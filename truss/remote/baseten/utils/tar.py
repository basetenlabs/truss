import tarfile
import tempfile
from pathlib import Path
from typing import IO, Any, Callable, List

from rich.progress import Progress
from truss.util.path import is_ignored


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


def create_tar_with_progress_bar(
    source_dir: Path, ignore_patterns: List[str] = [], delete=True
):
    # Exclude files that match the ignore_patterns
    files_to_include = [
        f
        for f in source_dir.rglob("*")
        if f.is_file() and not is_ignored(f, ignore_patterns, source_dir)
    ]

    total_size = sum(f.stat().st_size for f in files_to_include)
    temp_file = tempfile.NamedTemporaryFile(suffix=".tgz", delete=delete)

    with tarfile.open(temp_file.name, "w:") as tar:
        progress = Progress()

        task_id = progress.add_task("[cyan]Compressing...", total=total_size)

        with progress:

            def file_read_progress_callback(bytes_read: int):
                progress.update(task_id, advance=bytes_read)

            for file_path in files_to_include:
                arcname = str(file_path.relative_to(source_dir))
                with file_path.open("rb") as file_obj:
                    file_obj_with_progress = ReadProgressIndicatorFileHandle(
                        file_obj, file_read_progress_callback
                    )
                    tarinfo = tar.gettarinfo(name=str(file_path), arcname=arcname)
                    tar.addfile(
                        tarinfo=tarinfo,
                        fileobj=file_obj_with_progress,  # type: ignore[arg-type]
                    )
    return temp_file
