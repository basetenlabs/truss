import enum
import pathlib
from typing import List, Optional, Union

import pydantic

from truss.base import truss_config
from truss.base.errors import TrussUsageError
from truss.base.pydantic_models import SafeModel, SafeModelNonSerializable

LocalPath = Union[str, pathlib.Path]


class BasetenImage(enum.Enum):
    """Default images, curated by baseten, for different python versions. If a Chainlet
    uses GPUs, drivers will be included in the image."""

    # Enum values correspond to truss canonical python versions.
    PY39 = "py39"
    PY310 = "py310"
    PY311 = "py311"


class CustomImage(SafeModel):
    """Configures the usage of a custom image hosted on dockerhub."""

    image: str
    python_executable_path: Optional[str] = None
    docker_auth: Optional[truss_config.DockerAuthSettings] = None


class FileBundle(SafeModelNonSerializable):
    """A bundle of files to be copied into the docker image."""

    source_path: LocalPath
    remote_path: str


class DockerImage(SafeModelNonSerializable):
    """Configures the docker image in which a remoted chainlet is deployed.

    Note:
        Any paths are relative to the source file where ``DockerImage`` is
        defined and must be created with the helper function ``make_abs_path_here``.
        This allows you for example organize chainlets in different (potentially nested)
        modules and keep their requirement files right next their python source files.

    Args:
        base_image: The base image used by the chainlet. Other dependencies and
          assets are included as additional layers on top of that image. You can choose
          a Baseten default image for a supported python version (e.g.
          ``BasetenImage.PY311``), this will also include GPU drivers if needed, or
          provide a custom image (e.g. ``CustomImage(image="python:3.11-slim")``)..
        pip_requirements_file: Path to a file containing pip requirements. The file
          content is naively concatenated with ``pip_requirements``.
        pip_requirements: A list of pip requirements to install.  The items are
          naively concatenated with the content of the ``pip_requirements_file``.
        apt_requirements: A list of apt requirements to install.
        data_dir: Data from this directory is copied into the docker image and
          accessible to the remote chainlet at runtime.
        file_bundles: A list of directories containing additional python
          packages outside the chain's workspace dir, e.g. a shared library. This code
          is copied into the docker image and importable at runtime.
    """

    base_image: Union[BasetenImage, CustomImage] = BasetenImage.PY311
    pip_requirements_file: Optional[LocalPath] = None
    pip_requirements: List[str] = []
    apt_requirements: List[str] = []
    data_dir: Optional[LocalPath] = None
    file_bundles: Optional[List[FileBundle]] = None

    @pydantic.root_validator(pre=True)
    def migrate_fields(cls, values):
        if "base_image" in values:
            base_image = values["base_image"]
            if isinstance(base_image, str):
                doc_link = "https://docs.baseten.co/chains-reference/sdk#class-truss-chains-dockerimage"
                raise TrussUsageError(
                    "`DockerImage.base_image` as string is deprecated. Specify as "
                    f"`BasetenImage` or `CustomImage` (see docs: {doc_link})."
                )
        return values


class SecretReference(SafeModel):
    name: str

    def dict(self, **kwargs):
        return {"name": self.name, "type": "secret"}


class EnvironmentVariable(SafeModel):
    name: str
    value: Union[str, SecretReference]

    def dict(self, **kwargs):
        if isinstance(self.value, SecretReference):
            return {"name": self.name, "value": self.value.dict()}
        return {"name": self.name, "value": self.value}


class ImageSpec(SafeModel):
    name: str
    docker_image: DockerImage
    build_envvars: List[EnvironmentVariable] = []
    image_tag: Optional[str] = None
    build_commands: List[str] = []
    truss_version: Optional[str] = None
