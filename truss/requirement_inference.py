from dataclasses import dataclass

import pkg_resources


@dataclass
class Dependency:
    name: str
    version: str = None

    def __repr__(self):
        return f"{self.name}=={self.version}"


def infer_execution_requirements():
    ignored_packages = {"truss", "pip", "wheel", "setuptools", "pkg-resources"}
    # Gather all packages that are requirements and will be auto-installed.
    distributions = {}
    dependencies = set({})

    for distribution in pkg_resources.working_set:
        if distribution.key in ignored_packages:
            continue

        if distribution.key not in dependencies:
            distributions[distribution.key] = Dependency(
                distribution.key, distribution.version
            )

        for requirement in distribution.requires():
            if requirement.key not in ignored_packages:
                dependencies.add(requirement.key)

            if requirement.key in distributions:
                distributions.pop(requirement.key)

    return list(distributions.values())


if __name__ == "__main__":
    from pprint import pprint

    d0 = infer_execution_requirements()
    pprint(d0)
