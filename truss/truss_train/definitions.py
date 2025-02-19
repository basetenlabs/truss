# Training types here
import pathlib
from typing import Dict, Union

from truss.base.pydantic_models import SafeModel, SafeModelNonSerializable

LocalPath = Union[str, pathlib.Path]


class FileBundle(SafeModelNonSerializable):
    """A bundle of files to be copied into the docker image."""

    source_path: LocalPath
    remote_path: str


class SecretReference(SafeModel):
    name: str

    def dict(self, **kwargs):
        return {"name": self.name, "type": "secret"}


class EnvironmentVariables(SafeModelNonSerializable):
    """A dictionary for environment variables that can contain strings or secret references."""

    __root__: Dict[str, Union[str, SecretReference]]

    def dict(self):
        return {
            k: v.dict() if isinstance(v, SecretReference) else v
            for k, v in self.__root__.items()
        }
