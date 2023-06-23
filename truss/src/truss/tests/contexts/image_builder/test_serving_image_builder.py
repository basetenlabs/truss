from pathlib import Path
from tempfile import TemporaryDirectory

from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)
from truss.truss_handle import TrussHandle

BASE_DIR = Path(__file__).parent


def test_serving_image_dockerfile_from_user_base_image(custom_model_truss_dir):
    th = TrussHandle(custom_model_truss_dir)
    th.set_base_image("baseten/truss-server-base:3.9-v0.4.3", "/usr/local/bin/python3")
    builder_context = ServingImageBuilderContext
    image_builder = builder_context.run(th.spec.truss_dir)
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_builder.prepare_image_build_dir(tmp_path)
        with open(tmp_path / "Dockerfile", "r") as f:
            gen_docker_lines = f.readlines()
        with open(
            f"{BASE_DIR}/../../../test_data/server.Dockerfile",
            "r",
        ) as f:
            server_docker_lines = f.readlines()

        def filter_empty_lines(lines):
            return list(filter(lambda x: x and x != "\n", lines))

        gen_docker_lines = filter_empty_lines(gen_docker_lines)
        server_docker_lines = filter_empty_lines(server_docker_lines)
        assert gen_docker_lines == server_docker_lines
