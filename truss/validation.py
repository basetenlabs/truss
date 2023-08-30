import re
from pathlib import Path
from typing import Pattern, Set

from truss.constants import REGISTRY_BUILD_SECRET_PREFIX
from truss.errors import ValidationError

SECRET_NAME_MATCH_REGEX: Pattern[str] = re.compile(r"^[-._a-zA-Z0-9]+$")
MILLI_CPU_REGEX: Pattern[str] = re.compile(r"^[0-9.]*m$")
MEMORY_REGEX: Pattern[str] = re.compile(r"^[0-9.]*(\w*)$")
MEMORY_UNITS: Set[str] = set(
    ["k", "M", "G", "T", "P", "E", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei"]
)


def validate_secret_name(secret_name: str) -> None:
    if secret_name is None or not isinstance(secret_name, str) or secret_name == "":
        raise ValueError(f"Invalid secret name `{secret_name}`")

    def constraint_violation_msg():
        return f"Constraint violation for {secret_name}"

    if len(secret_name) > 253:
        raise ValueError(
            f"Secret name `{secret_name}` is longer than max allowed 253 chars."
        )

    if not SECRET_NAME_MATCH_REGEX.match(secret_name) and not secret_name.startswith(
        REGISTRY_BUILD_SECRET_PREFIX
    ):
        raise ValueError(
            constraint_violation_msg() + ", invalid characters found in secret name."
        )

    if secret_name == ".":
        raise ValueError(constraint_violation_msg() + ", secret name cannot be `.`")

    if secret_name == "..":
        raise ValueError(constraint_violation_msg() + ", secret name cannot be `..`")


def validate_cpu_spec(cpu_spec: str) -> None:
    if not isinstance(cpu_spec, str):
        raise ValidationError(
            f"{cpu_spec} needs to be a string, but is {type(cpu_spec)}"
        )

    if _is_numeric(cpu_spec):
        return

    is_milli_cpu_format = MILLI_CPU_REGEX.search(cpu_spec) is not None
    if not is_milli_cpu_format:
        raise ValidationError(f"Invalid cpu specification {cpu_spec}")


def validate_memory_spec(mem_spec: str) -> None:
    if not isinstance(mem_spec, str):
        raise ValidationError(
            f"{mem_spec} needs to be a string, but is {type(mem_spec)}"
        )
    if _is_numeric(mem_spec):
        return

    match = MEMORY_REGEX.search(mem_spec)
    if match is None:
        raise ValidationError(f"Invalid memory specification {mem_spec}")

    unit = match.group(1)
    if unit not in MEMORY_UNITS:
        raise ValidationError(f"Invalid memory unit {unit} in {mem_spec}")


def _is_numeric(number_like: str) -> bool:
    try:
        float(number_like)
        return True
    except ValueError:
        return False


def validate_python_executable_path(path: str) -> None:
    """
    This python executable path determines the python executable
    used to run the inference server - check to see that it is an absolute path
    """
    if path and not Path(path).is_absolute():
        raise ValidationError(
            f"Invalid relative python executable path {path}. Provide an absolute path"
        )
