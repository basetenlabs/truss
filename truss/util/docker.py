import enum
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from python_on_whales.components.container.cli_wrapper import Container

from truss.base.constants import TRUSS, TRUSS_DIR
from truss.local.local_config_handler import LocalConfigHandler


class Docker:
    _client = None

    @staticmethod
    def client():
        if Docker._client is None:
            from python_on_whales import DockerClient, docker

            if LocalConfigHandler.get_config().use_sudo:
                Docker._client = DockerClient(client_call=["sudo", "docker"])
            else:
                Docker._client = docker
        return Docker._client


def get_containers(labels: Dict, all: bool = False) -> list["Container"]:
    """Gets containers given labels."""
    return Docker.client().container.list(
        filters=_create_label_filters(labels), all=all
    )


def get_images(labels: Dict):
    """Gets images given labels."""
    return Docker.client().image.list(filters=_create_label_filters(labels))


def get_urls_from_container(container: Union[str, "Container"]) -> Dict[int, List[str]]:
    """Gets url where docker container is hosted."""
    if isinstance(container, str):
        container_inst = Docker.client().container.inspect(container)
    else:
        container_inst = container

    if (
        container_inst.network_settings is None
        or container_inst.network_settings.ports is None
    ):
        return {}
    ports = container_inst.network_settings.ports

    def parse_port(port_protocol) -> int:
        return int(port_protocol.split("/")[0])

    def url_from_port_protocol_value(port_protocol_value: Dict[str, str]) -> str:
        return (
            "http://"
            + port_protocol_value["HostIp"]
            + ":"
            + port_protocol_value["HostPort"]
        )

    def urls_from_port_protocol_values(
        port_protocol_values: List[Dict[str, str]],
    ) -> List[str]:
        return [url_from_port_protocol_value(v) for v in port_protocol_values]

    return {
        parse_port(port_protocol): urls_from_port_protocol_values(value)
        for port_protocol, value in ports.items()
        if value is not None
    }


def kill_containers(labels: Dict[str, bool]) -> None:
    from python_on_whales.exceptions import DockerException

    containers = get_containers(labels)
    for container in containers:
        container_labels = container.config.labels
        assert container_labels
        if TRUSS_DIR in container_labels:
            truss_dir = container_labels[TRUSS_DIR]
            logging.info(f"Killing Container: {container.id} for {truss_dir}")
    try:
        Docker.client().container.kill(containers)
    except DockerException:
        # The container may have stopped running by this point, this path
        # is for catching that. Unfortunately, there's no separate exception
        # for this scenario, so we catch the general one. Specific exceptions
        # such as NoSuchContainer are still allowed to error out.
        pass


def get_container_logs(container, follow, stream):
    return Docker.client().container.logs(container, follow=follow, stream=stream)


class DockerStates(enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    OOMKILLED = "oomkilled"
    DEAD = "dead"
    EXITED = "exited"


def inspect_container(container) -> "Container":
    """Inspects truss container"""
    return Docker.client().container.inspect(container)


def get_container_state(container) -> DockerStates:
    "Get state of the container"
    return DockerStates(inspect_container(container).state.status)


def _create_label_filters(labels: Dict) -> Dict[str, Any]:
    return {
        f"label={label_key}": label_value for label_key, label_value in labels.items()
    }


def kill_all() -> None:
    kill_containers({TRUSS: True})
