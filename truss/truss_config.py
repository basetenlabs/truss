from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from truss.types import ModelFrameworkType
from truss.validation import (
    validate_cpu_spec,
    validate_memory_spec,
    validate_secret_name,
)

DEFAULT_MODEL_FRAMEWORK_TYPE = ModelFrameworkType.CUSTOM
DEFAULT_MODEL_TYPE = "Model"
DEFAULT_MODEL_MODULE_DIR = "model"
DEFAULT_BUNDLED_PACKAGES_DIR = "packages"
DEFAULT_MODEL_CLASS_FILENAME = "model.py"
DEFAULT_MODEL_CLASS_NAME = "Model"
DEFAULT_TRUSS_STRUCTURE_VERSION = "2.0"
DEFAULT_MODEL_INPUT_TYPE = "Any"
DEFAULT_PYTHON_VERSION = "py39"
DEFAULT_DATA_DIRECTORY = "data"
DEFAULT_EXAMPLES_FILENAME = "examples.yaml"
DEFAULT_SPEC_VERSION = "2.0"
DEFAULT_SPEC_VERSION_ON_LOAD = "1.0"

DEFAULT_CPU = "500m"
DEFAULT_MEMORY = "512Mi"
DEFAULT_USE_GPU = False

DEFAULT_TRAINING_CLASS_FILENAME = "train.py"
DEFAULT_TRAINING_CLASS_NAME = "Train"
DEFAULT_TRAINING_MODULE_DIR = "train"


@dataclass
class Resources:
    cpu: str = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    use_gpu: bool = DEFAULT_USE_GPU

    @staticmethod
    def from_dict(d):
        cpu = d.get("cpu", DEFAULT_CPU)
        validate_cpu_spec(cpu)
        memory = d.get("memory", DEFAULT_MEMORY)
        validate_memory_spec(memory)

        return Resources(
            cpu=cpu,
            memory=memory,
            use_gpu=d.get("use_gpu", DEFAULT_USE_GPU),
        )

    def to_dict(self):
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "use_gpu": self.use_gpu,
        }


@dataclass
class Train:
    training_class_filename: str = DEFAULT_TRAINING_CLASS_FILENAME
    training_class_name: str = DEFAULT_TRAINING_CLASS_NAME
    training_module_dir: str = DEFAULT_TRAINING_MODULE_DIR
    variables: dict = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)

    @staticmethod
    def from_dict(d):
        return Train(
            training_class_filename=d.get(
                "training_class_filename", DEFAULT_TRAINING_CLASS_FILENAME
            ),
            training_class_name=d.get(
                "training_class_name", DEFAULT_TRAINING_CLASS_NAME
            ),
            training_module_dir=d.get(
                "training_module_dir", DEFAULT_TRAINING_MODULE_DIR
            ),
            variables=d.get("variables", {}),
            resources=Resources.from_dict(d.get("resources", {})),
        )

    def to_dict(self):
        return {
            "training_class_filename": self.training_class_filename,
            "training_class_name": self.training_class_name,
            "training_module_dir": self.training_module_dir,
            "variables": self.variables,
            "resources": self.resources.to_dict(),
        }


@dataclass
class TrussConfig:
    model_framework: ModelFrameworkType = DEFAULT_MODEL_FRAMEWORK_TYPE
    model_type: str = DEFAULT_MODEL_TYPE
    model_name: str = None

    model_module_dir: str = DEFAULT_MODEL_MODULE_DIR
    model_class_filename: str = DEFAULT_MODEL_CLASS_FILENAME
    model_class_name: str = DEFAULT_MODEL_CLASS_NAME

    data_dir: str = DEFAULT_DATA_DIRECTORY

    # Python types for what the model expects as input
    input_type: str = DEFAULT_MODEL_INPUT_TYPE
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    system_packages: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    python_version: str = DEFAULT_PYTHON_VERSION
    examples_filename: str = DEFAULT_EXAMPLES_FILENAME
    secrets: Dict[str, str] = field(default_factory=dict)
    description: str = None
    bundled_packages_dir: str = DEFAULT_BUNDLED_PACKAGES_DIR
    external_package_dirs: List[str] = field(default_factory=list)
    live_reload: bool = False
    # spec_version is a version string
    spec_version: str = DEFAULT_SPEC_VERSION
    train: Train = field(default_factory=Train)

    @property
    def canonical_python_version(self) -> str:
        return {
            "py39": "3.9",
            "py38": "3.8",
            "py37": "3.7",
        }[self.python_version]

    @staticmethod
    def from_dict(d):
        config = TrussConfig(
            # Users that are calling `load` on an existing Truss
            # should default to 1.0 whereas users creating a new Truss
            # should default to 2.0.
            spec_version=d.get("spec_version", DEFAULT_SPEC_VERSION_ON_LOAD),
            model_type=d.get("model_type", DEFAULT_MODEL_TYPE),
            model_framework=ModelFrameworkType(
                d.get("model_framework", DEFAULT_MODEL_FRAMEWORK_TYPE.value)
            ),
            model_module_dir=d.get("model_module_dir", DEFAULT_MODEL_MODULE_DIR),
            model_class_filename=d.get(
                "model_class_filename", DEFAULT_MODEL_CLASS_FILENAME
            ),
            model_class_name=d.get("model_class_name", DEFAULT_MODEL_CLASS_NAME),
            data_dir=d.get("data_dir", DEFAULT_DATA_DIRECTORY),
            input_type=d.get("input_type", DEFAULT_MODEL_INPUT_TYPE),
            model_metadata=d.get("model_metadata", {}),
            requirements=d.get("requirements", []),
            system_packages=d.get("system_packages", []),
            environment_variables=d.get("environment_variables", {}),
            resources=Resources.from_dict(d.get("resources", {})),
            python_version=d.get("python_version", DEFAULT_PYTHON_VERSION),
            model_name=d.get("model_name", None),
            examples_filename=d.get("examples_filename", DEFAULT_EXAMPLES_FILENAME),
            secrets=d.get("secrets", {}),
            description=d.get("description", None),
            bundled_packages_dir=d.get(
                "bundled_packages_dir", DEFAULT_BUNDLED_PACKAGES_DIR
            ),
            external_package_dirs=d.get("external_package_dirs", []),
            live_reload=d.get("live_reload", False),
            train=Train.from_dict(d.get("train", {})),
        )
        config.validate()
        return config

    @staticmethod
    def from_yaml(yaml_path: Path):
        with yaml_path.open() as yaml_file:
            return TrussConfig.from_dict(yaml.safe_load(yaml_file))

    def write_to_yaml_file(self, path: Path):
        with path.open("w") as config_file:
            yaml.dump(self.to_dict(), config_file)

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "model_framework": self.model_framework.value,
            "model_module_dir": self.model_module_dir,
            "model_class_filename": self.model_class_filename,
            "model_class_name": self.model_class_name,
            "data_dir": self.data_dir,
            "input_type": self.input_type,
            "model_metadata": self.model_metadata,
            "requirements": self.requirements,
            "system_packages": self.system_packages,
            "environment_variables": self.environment_variables,
            "resources": self.resources.to_dict(),
            "python_version": self.python_version,
            "model_name": self.model_name,
            "examples_filename": self.examples_filename,
            "secrets": self.secrets,
            "description": self.description,
            "bundled_packages_dir": self.bundled_packages_dir,
            "external_package_dirs": self.external_package_dirs,
            "live_reload": self.live_reload,
            "spec_version": self.spec_version,
            "train": self.train.to_dict(),
        }

    def clone(self):
        return TrussConfig.from_dict(self.to_dict())

    def validate(self):
        for secret_name in self.secrets:
            validate_secret_name(secret_name)
