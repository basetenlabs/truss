from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from truss.contexts.local_loader.dockerfile_parser import (
    DockerInstruction,
    parse_dockerfile,
)
from truss.util.path import copy_tree_or_file


@dataclass
class DockerBuildEmulatorResult:
    workdir: Path = field(default_factory=lambda: Path("/"))
    env: Dict = field(default_factory=dict)
    entrypoint: List = field(default_factory=list)


class DockerBuildEmulator:
    """Emulates Docker Builds

    As running docker builds is expensive, this class emulates the docker build
    by parsing the docker file and applying certain commands to create an
    appropriate enviroment in a directory to simulate the root of the file system.

    Support COPY, ENV, ENTRYPOINT, WORKDIR commands. All other commands are ignored.
    """

    def __init__(self, dockerfile_path: Path, context_dir: Path) -> None:
        self._commands = parse_dockerfile(dockerfile_path)
        self._context_dir = context_dir

    def run(self, fs_root_dir: Path) -> DockerBuildEmulatorResult:
        def _resolve_env(key: str) -> str:
            if key.startswith("$"):
                key = key.replace("$", "", 1)
                v = result.env[key]
                return v
            return key

        def _resolve_values(keys: List[str]) -> List[str]:
            return list(map(_resolve_env, keys))

        result = DockerBuildEmulatorResult()
        for cmd in self._commands:
            if not cmd.is_supported:
                continue
            values = _resolve_values(cmd.value)
            if cmd.instruction == DockerInstruction.ENV:
                if "=" in values[0]:
                    values = values[0].split("=", 1)
                result.env[values[0]] = values[1]
            if cmd.instruction == DockerInstruction.ENTRYPOINT:
                result.entrypoint = list(values)
            if cmd.instruction == DockerInstruction.COPY:
                src, dst = values
                src = src.replace("./", "", 1)
                dst = dst.replace("/", "", 1)
                copy_tree_or_file(self._context_dir / src, fs_root_dir / dst)
            if cmd.instruction == DockerInstruction.WORKDIR:
                result.workdir = result.workdir / values[0]
        return result
