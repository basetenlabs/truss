import pytest
from python_on_whales import docker
from truss.docker import get_urls_from_container


@pytest.fixture
def docker_container():
    container = docker.container.create("nginx", publish=[[8080, 8080]])
    try:
        container.start()
        yield container
    finally:
        container.stop()


@pytest.mark.integration
def test_get_urls_from_container(docker_container):
    print(get_urls_from_container(docker_container))
