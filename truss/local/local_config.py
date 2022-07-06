from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class LocalConfig:
    secrets: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(d):
        return LocalConfig(
            secrets=d.get('secrets', {}),
        )

    @staticmethod
    def from_yaml(yaml_path: Path):
        with yaml_path.open() as yaml_file:
            return LocalConfig.from_dict(yaml.safe_load(yaml_file))

    def to_dict(self):
        return {
            'secrets': self.secrets,
        }

    def write_to_yaml_file(self, path: Path):
        with path.open('w') as config_file:
            yaml.dump(self.to_dict(), config_file)
