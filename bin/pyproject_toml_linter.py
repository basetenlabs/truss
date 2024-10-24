# type: ignore
import collections
import pathlib

import tomlkit


def populate_extras(pyproject_path: pathlib.Path) -> None:
    with pyproject_path.open("r") as f:
        content = tomlkit.parse(f.read())

    dependencies = content["tool"]["poetry"]["dependencies"]
    dependency_metadata = content["tool"]["dependency_metadata"]

    extra_sections = collections.defaultdict(set)
    all_deps = set()
    for key in dependencies.keys():
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
                f"`{key}` in `[tool.dependency_metadata]` which is not in "
                "`[tool.poetry.dependencies]` "
                f"(file: {pyproject_path}). please remove / sync."
            )
    extras_section = tomlkit.table()
    for extra_section, deps in extra_sections.items():
        extras_section[extra_section] = tomlkit.array()
        extras_section[extra_section].extend(sorted(deps))

    extras_section["all"] = tomlkit.array()
    extras_section["all"].extend(sorted(all_deps))

    assert content["tool"]["poetry"]["extras"]
    content["tool"]["poetry"]["extras"] = extras_section

    with pyproject_path.open("w") as f:
        f.write(tomlkit.dumps(content))


if __name__ == "__main__":
    pyproject_file = pathlib.Path(__file__).parent.parent.resolve() / "pyproject.toml"
    populate_extras(pyproject_file)
