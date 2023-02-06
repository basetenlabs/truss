from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml
from truss.truss_config import (
    DEFAULT_MODEL_FRAMEWORK_TYPE,
    DEFAULT_PYTHON_VERSION,
    ModelFrameworkType,
    Resources,
)


@dataclass
class MultiTrussConfig:
    # Relative Path's to all the member trusses
    # TODO: update to be object with names
    trusses: List[str] = field(default_factory=list)
    python_version: str = DEFAULT_PYTHON_VERSION
    resources: Resources = field(default_factory=Resources)
    model_framework: ModelFrameworkType = DEFAULT_MODEL_FRAMEWORK_TYPE
    environment_variables: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_yaml(yaml_path: Path):
        with yaml_path.open() as yaml_file:
            return MultiTrussConfig.from_dict(yaml.safe_load(yaml_file))

    def write_to_yaml_file(self, path: Path):
        with path.open("w") as config_file:
            yaml.dump(self.to_dict(), config_file)

    def to_dict(self):
        return {
            "trusses": self.trusses,
            "python_version": self.python_version,
            "resources": self.resources.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        config = MultiTrussConfig(
            trusses=d.get("trusses", []),
            python_version=d.get("python_version", DEFAULT_PYTHON_VERSION),
            resources=Resources.from_dict(d.get("resources", {})),
        )
        config.validate()
        return config

    def clone(self):
        return MultiTrussConfig.from_dict(self.to_dict())

    def validate(self):
        if len(self.trusses) < 2:
            raise ValueError("MultiTruss is only useful is you have at least 2 models")

    @property
    def canonical_python_version(self) -> str:
        return {
            "py39": "3.9",
            "py38": "3.8",
            "py37": "3.7",
        }[self.python_version]
