from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from truss.util.docker import Docker
from truss.util.path import given_or_temporary_dir


class ImageBuilder(ABC):
    def build_image(
        self,
        build_dir: Optional[Path] = None,
        tag: Optional[str] = None,
        labels: Optional[dict] = None,
        cache: bool = True,
        network: Optional[str] = None,
    ):
        """Build image.

        Arguments:
            build_dir(Path): Directory to use for building the docker image. If None
                             then a temporary directory is used.
            tag(str): A tag to assign to the docker image.
        """

        with given_or_temporary_dir(build_dir) as build_dir_path:
            self.prepare_image_build_dir(build_dir_path)
            return Docker.client().build(
                str(build_dir_path),
                labels=labels if labels else {},
                tags=tag or self.default_tag,
                cache=cache,
                network=network,
                load=True,  # We pass load=True so that `build` returns an `Image` object
            )

    @property
    @abstractmethod
    def default_tag(self):
        pass

    @abstractmethod
    def prepare_image_build_dir(self, build_dir: Optional[Path] = None):
        pass

    def docker_build_command(self, build_dir) -> str:
        return f"docker build {build_dir} -t {self.default_tag}"
