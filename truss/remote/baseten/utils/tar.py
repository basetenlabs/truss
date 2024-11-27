import contextlib
import tarfile
import tempfile
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type

if TYPE_CHECKING:
    from rich import progress

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
    source_dir: Path,
    ignore_patterns: Optional[List[str]] = None,
    delete=True,
    progress_bar: Optional[Type["progress.Progress"]] = None,
):
    files_to_include = [
        f
        for f in source_dir.rglob("*")
        if f.is_file() and not is_ignored(f, ignore_patterns or [], source_dir)
    ]

    total_size = sum(f.stat().st_size for f in files_to_include)
    temp_file = tempfile.NamedTemporaryFile(suffix=".tgz", delete=delete)

    progress_context = (
        progress_bar(transient=True) if progress_bar else contextlib.nullcontext()
    )
    # Trailing spaces are to align with `multipart_upload_boto3` message.
    task_id = (
        progress_context.add_task("[cyan]Packing Truss  ", total=total_size)
        if not isinstance(progress_context, contextlib.nullcontext)
        else None
    )

    def file_read_progress_callback(bytes_read: int):
        if not isinstance(progress_context, contextlib.nullcontext):
            assert task_id is not None
            progress_context.update(task_id, advance=bytes_read)

    with tarfile.open(temp_file.name, "w:") as tar, progress_context:
        for file_path in files_to_include:
            arcname = str(file_path.relative_to(source_dir))
            with file_path.open("rb") as file_obj:
                file_obj_with_progress = (
                    ReadProgressIndicatorFileHandle(
                        file_obj, file_read_progress_callback
                    )
                    if progress_bar
                    else file_obj
                )
                tarinfo = tar.gettarinfo(name=str(file_path), arcname=arcname)
                tar.addfile(tarinfo=tarinfo, fileobj=file_obj_with_progress)  # type: ignore[arg-type]  # `ReadProgressIndicatorFileHandle` implements `IO[bytes]`.
    return temp_file
