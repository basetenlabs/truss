import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError

import yaml
from truss.constants import CONFIG_FILE, INFERENCE_SERVER_PORT, TRUSS_DIR
from truss.contexts.image_builder.multi_truss_image_builder import (
    MultiTrussImageBuilderContext,
)
from truss.docker import Docker, kill_containers
from truss.errors import ContainerIsDownError, ContainerNotFoundError
from truss.truss_handle import TrussHandle, wait_for_truss

logger = logging.getLogger(__name__)


@dataclass
class MultiTrussConfig:
    # Relative Path's to all the member trusses
    trusses: List[str] = field(default_factory=list)

    @staticmethod
    def from_yaml(yaml_path: Path):
        with yaml_path.open() as yaml_file:
            return MultiTrussConfig.from_dict(yaml.safe_load(yaml_file))

    def write_to_yaml_file(self, path: Path):
        with path.open("w") as config_file:
            yaml.dump(self.to_dict(), config_file)

    def to_dict(self):
        return {"trusses": self.trusses}

    @staticmethod
    def from_dict(d):
        config = MultiTrussConfig(trusses=d.get("trusses", []))
        config.validate()
        return config

    def clone(self):
        return MultiTrussConfig.from_dict(self.to_dict())

    def validate(self):
        if len(self.trusses) < 2:
            raise ValueError("MultiTruss is only useful is you have at least 2 models")


class MultiTrussSpec:
    def __init__(self, multi_truss_dir: Path) -> None:
        self.dir = multi_truss_dir
        self.config = MultiTrussConfig.from_yaml(multi_truss_dir / CONFIG_FILE)

    @property
    def trusses_dir_paths(self) -> List[Path]:
        paths = []
        for path_name in self.config.trusses:
            path = Path(path_name)
            if path.is_absolute():
                paths.append(path)
            else:
                paths.append(self.dir / path)
        return paths

    @property
    def prepared_truss_dir_paths(self) -> List[Path]:
        # Make sure that all the children trusses are ready to be copied
        return list(
            [
                TrussHandle(truss_path, validate=True).gather()
                for truss_path in self.trusses_dir_paths
            ]
        )


class MultiTrussHandle:
    def __init__(self, multi_truss_dir: Path) -> None:
        self._dir = multi_truss_dir
        self._spec = MultiTrussSpec(self._dir)

    @property
    def spec(self) -> MultiTrussSpec:
        return self._spec

    def _build_image(
        self,
        builder_context,
        labels: Dict[str, str],
        build_dir: Path = None,
        tag: str = None,
    ):

        # TODO: handle image rebuilds with labels
        # image = _docker_image_from_labels(labels=labels)
        # if image is not None:
        #     return image

        build_dir_path = Path(build_dir) if build_dir is not None else None
        image_builder = builder_context.run(self._dir)
        build_image_result = image_builder.build_image(
            build_dir_path,
            tag,
            labels=labels,
        )
        return build_image_result

    def build_serving_docker_image(self, build_dir: Path = None, tag: str = None):

        image = self._build_image(
            builder_context=MultiTrussImageBuilderContext,
            # TODO: add real labels
            labels={TRUSS_DIR: self._dir},
            build_dir=build_dir,
            tag=tag,
        )
        # TODO: add signature
        # self._store_signature()
        return image

    def kill_container(self):
        """Kill container

        Killing is done based on directory of the truss.
        """
        kill_containers({TRUSS_DIR: self._dir})

    def docker_run(
        self,
        build_dir: Path = None,
        tag: str = None,
        local_port: int = INFERENCE_SERVER_PORT,
        detach=True,
    ):
        """
        Builds a docker image and runs it as a container.

        Args:
            build_dir: Directory to use for creating docker build context.
            tag: Tags to apply to docker image.
            local_port: Local port to forward inference server to.
            detach: Run docker container in detached mode.

        Returns:
            Container, which can be used to get information about the running,
            including its id. The id can be used to kill the container.
        """
        image = self.build_serving_docker_image(build_dir=build_dir, tag=tag)
        # TODO: Handle secrets
        # secrets_mount_dir_path = _prepare_secrets_mount_dir()
        publish_ports = [[local_port, INFERENCE_SERVER_PORT]]

        # We are going to try running a new container, make sure previous one is gone
        self.kill_container()
        labels = {TRUSS_DIR: self._dir}

        envs = {}

        container = Docker.client().run(
            image.id,
            publish=publish_ports,
            detach=detach,
            labels=labels,
            # mounts=[
            #     [
            #         "type=bind",
            #         f"src={str(secrets_mount_dir_path)}",
            #         "target=/secrets",
            #     ]
            # ],
            # TODO: only require GPUs if needed
            gpus="all",  # if self._spec.config.resources.use_gpu else None,
            envs=envs,
            add_hosts=[("host.docker.internal", "host-gateway")],
        )
        logger.info(
            f"Multi Models server started on port {local_port}, docker container id {container.id}"
        )
        # TODO: handle waiting for startup. For now, just check liveness.
        model_base_url = f"http://localhost:{local_port}/"
        try:
            wait_for_truss(model_base_url, container)
        except ContainerNotFoundError as err:
            raise err
        except (ContainerIsDownError, HTTPError, ConnectionError) as err:
            # TODO: grab container logs
            logger.error(
                "Thinsg are broken"
            )  # self.serving_container_logs(follow=False, stream=False))
            raise err

        return container
