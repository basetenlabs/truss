from typing import List

from truss_chains.truss_chains.definitions import DockerImage, SafeModel


class Secret(SafeModel):
    name: str


class ImageSpec(SafeModel):
    name: str
    docker_image: DockerImage
    build_secrets: List[Secret]
