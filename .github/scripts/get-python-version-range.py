import os
import sys

import tomlkit
from packaging.specifiers import SpecifierSet
from packaging.version import Version

with open("pyproject.toml", "r") as f:
    doc = tomlkit.load(f)

constraint = doc["tool"]["poetry"]["dependencies"]["python"]  # type: ignore[index]
specifier = SpecifierSet(str(constraint))

candidates = [Version(f"3.{i}") for i in range(0, 20)]
matching = sorted([v for v in candidates if v in specifier])

if not matching:
    sys.exit("::error ::No Python versions match the constraint!")

output_path = os.environ["GITHUB_OUTPUT"]
with open(output_path, "a") as f:
    f.write(f"min_supported_python_version={matching[0]}\n")
    f.write(f"max_supported_python_version={matching[-1]}\n")
