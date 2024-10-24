import pytest
from python_on_whales import docker

from truss.util.docker import get_urls_from_container


@pytest.fixture
def docker_container():
    container = docker.container.create("nginx", publish=[[19051, 19051]])
    try:
        container.start()
        yield container
    finally:
        container.stop()


@pytest.mark.integration
def test_get_urls_from_container(docker_container):
    resp = get_urls_from_container(docker_container)
    assert resp == {19051: ["http://0.0.0.0:19051", "http://:::19051"]}
