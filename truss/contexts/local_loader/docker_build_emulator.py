import re
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
        def _resolve_env(in_value: str) -> str:
            # Valid environment variable name pattern
            var_name_pattern = r"[A-Za-z_][A-Za-z0-9_]*"

            # Handle ${VAR} syntax
            def replace_braced_var(match):
                var_name = match.group(1)
                return result.env.get(
                    var_name, match.group(0)
                )  # Return original if not found

            # Handle $VAR syntax (word boundary ensures we don't match parts of other vars)
            def replace_simple_var(match):
                var_name = match.group(1)
                return result.env.get(
                    var_name, match.group(0)
                )  # Return original if not found

            # Replace ${VAR} patterns first, using % substitution to avoid additional braces noise with f-strings
            value = re.sub(
                r"\$\{(%s)\}" % var_name_pattern, replace_braced_var, in_value
            )
            # Then replace remaining $VAR patterns (only at word boundaries)
            value = re.sub(r"\$(%s)\b" % var_name_pattern, replace_simple_var, value)

            return value

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
                # Filter out --chown flags
                filtered_values = [v for v in values if not v.startswith("--chown")]

                # NB(nikhil): Skip COPY commands with --from flag (multi-stage builds)
                if len(filtered_values) != 2:
                    continue

                src, dst = filtered_values
                src = src.replace("./", "", 1)
                dst = dst.replace("/", "", 1)
                copy_tree_or_file(self._context_dir / src, fs_root_dir / dst)
            if cmd.instruction == DockerInstruction.WORKDIR:
                result.workdir = result.workdir / values[0]
        return result
