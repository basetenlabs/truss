import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_build_docker_image():
    root_path = Path(__file__).parent.parent.parent
    root = str(root_path)
    context_builder_image_test_dir = str(
        root_path / "truss" / "test_data" / "context_builder_image_test"
    )

    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            ".",
            "-f",
            "context_builder.Dockerfile",
            "--platform=linux/amd64",
            "-t",
            "baseten/truss-context-builder:test",
        ],
        check=True,
        cwd=root,
    )

    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            context_builder_image_test_dir,
            "--platform=linux/amd64",
            "-t",
            "baseten/truss-context-builder-test",
        ],
        check=True,
        cwd=root,
    )

    # This will throw if building docker build context fails
    subprocess.run(
        [
            "docker",
            "run",
            "baseten/truss-context-builder-test",
        ],
        check=True,
        cwd=root,
    )
