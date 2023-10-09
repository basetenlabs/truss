import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from truss.constants import CONFIG_FILE
from truss.errors import ValidationError
from truss.truss_config import ExternalData, ModelServer, TrussConfig
from truss.types import Example, ModelFrameworkType


class TrussSpec:
    """
    Helper class for easy access to information in a Truss.
    """

    def __init__(self, truss_dir: Path) -> None:
        self._truss_dir = truss_dir
        self._config = TrussConfig.from_yaml(truss_dir / CONFIG_FILE)

    @property
    def truss_dir(self) -> Path:
        return self._truss_dir

    @property
    def config_path(self) -> Path:
        return self._truss_dir / CONFIG_FILE

    @property
    def data_dir(self) -> Path:
        return self._truss_dir / self._config.data_dir

    @property
    def external_data(self) -> Optional[ExternalData]:
        return self._config.external_data

    @property
    def model_module_dir(self) -> Path:
        return self._truss_dir / self._config.model_module_dir

    @property
    def training_module_dir(self) -> Path:
        return self._truss_dir / self._config.train.training_module_dir

    @property
    def bundled_packages_dir(self) -> Path:
        return self._truss_dir / self._config.bundled_packages_dir

    @property
    def external_package_dirs_paths(self) -> List[Path]:
        paths = []
        for path_name in self._config.external_package_dirs:
            path = Path(path_name)
            if path.is_absolute():
                paths.append(path)
            else:
                paths.append(self._truss_dir / path)
        return paths

    @property
    def model_class_filepath(self) -> Path:
        conf = self._config
        return self._truss_dir / conf.model_module_dir / conf.model_class_filename

    @property
    def train_class_filepath(self) -> Path:
        conf = self._config
        return self.training_module_dir / conf.train.training_class_filename

    @property
    def train_module_name(self) -> str:
        return str(Path(self._config.train.training_class_filename).with_suffix(""))

    @property
    def train_module_fullname(self) -> str:
        return f"{self._config.train.training_module_dir}.{self.train_module_name}"

    @property
    def train_class_name(self) -> str:
        return self._config.train.training_class_name

    @property
    def config(self) -> TrussConfig:
        return self._config

    @property
    def model_server(self) -> ModelServer:
        return self.config.build.model_server

    @property
    def spec_version(self) -> str:
        return self._config.spec_version

    @property
    def python_version(self) -> str:
        return self._config.python_version

    @property
    def canonical_python_version(self) -> str:
        return self._config.canonical_python_version

    @property
    def cpu(self) -> str:
        return self._config.resources.cpu

    @property
    def json_string(self) -> str:
        return json.dumps(self._config.to_dict())

    @property
    def memory(self) -> str:
        return self._config.resources.memory

    @property
    def use_gpu(self) -> str:
        return self._config.resources.use_gpu

    @property
    def model_module_name(self) -> str:
        return str(Path(self._config.model_class_filename).with_suffix(""))

    @property
    def model_module_fullname(self) -> str:
        return f"{self._config.model_module_dir}.{self.model_module_name}"

    @property
    def model_class_name(self) -> str:
        return self._config.model_class_name

    @property
    def model_framework_type(self) -> ModelFrameworkType:
        return self._config.model_framework

    @property
    def model_framework_name(self) -> str:
        return self.model_framework_type.value

    @property
    def requirements(self) -> List[str]:
        return self._config.requirements

    @property
    def requirements_txt(self) -> str:
        return _join_lines(self._config.requirements)

    @property
    def system_packages(self) -> List[str]:
        return self._config.system_packages

    @property
    def system_packages_txt(self) -> str:
        return _join_lines(self._config.system_packages)

    @property
    def environment_variables(self) -> Dict[str, str]:
        return self._config.environment_variables

    @property
    def examples_path(self) -> Path:
        return self._truss_dir / self._config.examples_filename

    @property
    def examples(self) -> List[Example]:
        with self.examples_path.open() as yaml_file:
            examples = yaml.safe_load(yaml_file)
            if examples is None:
                examples = []
            if not isinstance(examples, list):
                raise ValidationError(
                    f"Examples should be provided as a list but found to be {type(examples)}"
                )
            return [Example.from_dict(example) for example in examples]

    @property
    def yaml_string(self) -> str:
        with self.config_path.open() as yaml_file:
            return yaml_file.read()

    @property
    def secrets(self) -> Dict[str, str]:
        return self._config.secrets

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def live_reload(self) -> bool:
        return self._config.live_reload

    @property
    def base_image_name(self) -> Optional[str]:
        return self._config.base_image.image

    @property
    def python_executable_path(self) -> str:
        return self._config.base_image.python_executable_path

    @property
    def apply_library_patches(self) -> bool:
        return self._config.apply_library_patches

    @property
    def hash_ignore_patterns(self) -> List[str]:
        """By default, data directory contents are ignored when hashing,
        as patching is not supported for these changes.
        """
        return [f"{self.data_dir}/*"]


def _join_lines(lines: List[str]) -> str:
    return "\n".join(lines) + "\n"
