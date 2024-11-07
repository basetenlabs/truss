# type: ignore  # tomlkit APIs are messy.
import collections
import pathlib
from typing import DefaultDict, Set

import tomlkit


def _populate_extras(pyproject_path: pathlib.Path) -> None:
    with pyproject_path.open("r", encoding="utf-8") as file:
        original_content = file.read()
        content = tomlkit.parse(original_content)

    dependencies = content["tool"]["poetry"]["dependencies"]
    dependency_metadata = content["tool"]["dependency_metadata"]

    extra_sections: DefaultDict[str, Set[str]] = collections.defaultdict(set)
    all_deps: Set[str] = set()

    for key, value in dependencies.items():
        if isinstance(value, dict):
            is_optional = value.get("optional", False)
        else:
            is_optional = False  # Base dependencies.

        if not is_optional:
            continue

        if key not in dependency_metadata:
            raise ValueError(
                f"`{key}` is missing in `[tool.dependency_metadata]`. "
                f"(file: {pyproject_path}). Please add metadata."
            )
        metadata = dependency_metadata[key]
        components = metadata["components"].split(",")
        for component in components:
            if component == "base":
                continue
            extra_sections[component].add(key)
            all_deps.add(key)

    for key in dependency_metadata.keys():
        if key not in dependencies:
            raise ValueError(
                f"`{key}` in `[tool.dependency_metadata]` is not in "
                "`[tool.poetry.dependencies]`. "
                f"(file: {pyproject_path}). Please remove or sync."
            )

    extras_section = tomlkit.table()
    for extra_section, deps in extra_sections.items():
        extras_section[extra_section] = tomlkit.array()
        extras_section[extra_section].extend(sorted(deps))

    extras_section["all"] = tomlkit.array()
    extras_section["all"].extend(sorted(all_deps))

    if "extras" not in content["tool"]["poetry"]:
        raise ValueError("Expected section [tool.poetry.extras] to be present.")

    content["tool"]["poetry"]["extras"] = extras_section

    updated_content = tomlkit.dumps(content)

    # Compare the content before and after; if changes were made, fail the check
    if original_content != updated_content:
        with pyproject_path.open("w", encoding="utf-8") as file:
            file.write(updated_content)
        print(f"File '{pyproject_path}' was updated. Please re-stage the changes.")
        exit(1)

    print("No changes detected.")


if __name__ == "__main__":
    pyproject_file = pathlib.Path(__file__).parent.parent.resolve() / "pyproject.toml"
    _populate_extras(pyproject_file)
