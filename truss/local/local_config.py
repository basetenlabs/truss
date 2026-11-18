from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class LocalConfig:
    use_sudo: bool = False

    @staticmethod
    def from_dict(d):
        return LocalConfig(use_sudo=d.get("use_sudo", False))

    @staticmethod
    def from_yaml(yaml_path: Path):
        with yaml_path.open() as yaml_file:
            return LocalConfig.from_dict(yaml.safe_load(yaml_file))
