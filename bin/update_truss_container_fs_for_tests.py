#!/usr/bin/env python3

# This script generates truss/test_data/truss_container_fs from
# truss/test_data/test_truss.
#
# Idea is to get access to representative /app folder content of a sample truss
# for testing purposes.
#
# Contents of truss_container_fs should then be checked into git.

import subprocess
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.resolve())


if __name__ == "__main__":
    subprocess.run(["truss", "run-image", "truss/test_data/test_truss"], cwd=ROOT)
    ps_output = subprocess.check_output(
        [
            "docker",
            "ps",
            "--filter",
            "label=truss_dir=truss/test_data/test_truss",
            "--format",
            "'{{.Names}}'",
        ]
    )
    container_name = ps_output.decode("utf-8").strip()[1:-1]
    subprocess.run(
        [
            "docker",
            "cp",
            f"{container_name}:/app",
            "truss/test_data/truss_container_fs",
        ]
    )
    subprocess.run(
        [
            "docker",
            "kill",
            container_name,
        ]
    )
