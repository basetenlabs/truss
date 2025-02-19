# Training types here
import pathlib
from typing import Union

from truss.base.pydantic_models import SafeModelNonSerializable

LocalPath = Union[str, pathlib.Path]


class FileBundle(SafeModelNonSerializable):
    """A bundle of files to be copied into the docker image."""

    source_path: LocalPath
    remote_path: str
