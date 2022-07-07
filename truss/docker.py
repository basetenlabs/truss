import logging
from typing import Dict

from truss.constants import TRUSS_DIR
from truss.local.local_config_handler import LocalConfigHandler


class Docker:
    _client = None

    @staticmethod
    def client():
        if Docker._client is None:
            from python_on_whales import DockerClient, docker
            if LocalConfigHandler.get_config().use_sudo:
                Docker._client = DockerClient(client_call=['sudo', 'docker'])
            else:
                Docker._client = docker
        return Docker._client


def get_containers(labels: dict, all=False):
    """Gets containers given labels."""
    return Docker.client().container.list(filters=_create_label_filters(labels), all=all)


def get_images(labels: dict):
    """Gets images given labels."""
    return Docker.client().image.list(filters=_create_label_filters(labels))


def get_urls_from_container(container):
    """Gets url where docker container is hosted."""
    ports = container.network_settings.ports
    valid_urls = []
    for port in ports.values():
        for urls in port:
            valid_urls.append("http://" + urls["HostIp"] + ":" + urls["HostPort"])
    return valid_urls


def kill_containers(labels: Dict[str, str]):
    containers = get_containers(labels)
    for container in containers:
        container_labels = container.config.labels
        if TRUSS_DIR in container_labels:
            truss_dir = container_labels[TRUSS_DIR]
            logging.info(f"Killing Container: {container.id} for {truss_dir}")
    Docker.client().container.kill(containers)


def get_container_logs(container):
    return Docker.client().container.logs(container, follow=True, stream=True)


def _create_label_filters(labels: dict):
    return {
        f'label={label_key}': label_value
        for label_key, label_value in labels.items()
    }
