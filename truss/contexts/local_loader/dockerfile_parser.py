from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class DockerInstruction(Enum):
    COPY = "COPY"
    ENV = "ENV"
    ENTRYPOINT = "ENTRYPOINT"
    WORKDIR = "WORKDIR"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, instruction: str) -> "DockerInstruction":
        try:
            return cls(instruction.upper())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class DockerCommand:
    instruction: DockerInstruction
    value: List[str]

    @property
    def is_supported(self) -> bool:
        return self.instruction != DockerInstruction.UNKNOWN


def parse_dockerfile(dockerfile_path: Path) -> List[DockerCommand]:
    commands = []
    with open(dockerfile_path, "r") as f:
        content = f.read()

    # Join across line continuations.
    content = content.replace("\\\n", " ")
    for line in content.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split(" ", 1)
        if len(parts) == 2:
            instruction_str, args_str = parts
            instruction = DockerInstruction.from_string(instruction_str)
            args = args_str.split()
            commands.append(DockerCommand(instruction=instruction, value=args))

    return commands
