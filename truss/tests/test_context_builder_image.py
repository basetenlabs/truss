import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_build_docker_image():
    root_path = Path(__file__).parent.parent.parent
    root = str(root_path)
    context_builder_image_test_dir = str(
        root_path / "truss" / "tests" / "test_data" / "context_builder_image_test"
    )

    proc = subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            ".",
            "-f",
            "context_builder.Dockerfile",
            "--platform=linux/amd64",
            "-t",
            "truss-context-builder-for-testing",
        ],
        cwd=root,
        capture_output=True,
    )
    print(_proc_output_str(proc))  # todo remove
    if proc.returncode != 0:
        assert (
            False
        ), f"Failed to build context builder image :: {_proc_output_str(proc)}"

    proc = subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            context_builder_image_test_dir,
            "--platform=linux/amd64",
            "-t",
            "truss-context-builder-test",
        ],
        cwd=root,
        capture_output=True,
    )
    print(_proc_output_str(proc))  # todo remove
    if proc.returncode != 0:
        assert (
            False
        ), f"Failed to build context builder test image :: {_proc_output_str(proc)}"

    # This will throw if building docker build context fails
    proc = subprocess.run(
        [
            "docker",
            "run",
            "truss-context-builder-test",
        ],
        cwd=root,
        capture_output=True,
    )
    if proc.returncode != 0:
        assert (
            False
        ), f"Context builder test docker run failed :: {_proc_output_str(proc)}"


def _proc_output_str(proc) -> str:
    output = {}
    if proc.stderr is not None:
        output["stderr"] = proc.stderr.decode("utf-8")
    if proc.stdout is not None:
        output["stdout"] = proc.stdout.decode("utf-8")
    return json.dumps(output)
