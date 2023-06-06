from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from truss.constants import HTTP_PUBLIC_BLOB_BACKEND
from truss.errors import ValidationError
from truss.types import ModelFrameworkType
from truss.util.data_structures import transform_optional
from truss.validation import (
    validate_cpu_spec,
    validate_memory_spec,
    validate_python_executable_path,
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

DEFAULT_BLOB_BACKEND = HTTP_PUBLIC_BLOB_BACKEND


class Accelerator(Enum):
    T4 = "T4"
    A10G = "A10G"
    V100 = "V100"
    A100 = "A100"


@dataclass
class AcceleratorSpec:
    accelerator: Optional[Accelerator] = None
    count: int = 0

    def to_str(self) -> Optional[str]:
        if self.accelerator is None or self.count == 0:
            return None
        if self.count > 1:
            return f"{self.accelerator.value}:{self.count}"
        return self.accelerator.value

    @staticmethod
    def from_str(acc_spec: Optional[str]):
        if acc_spec is None:
            return AcceleratorSpec()
        parts = acc_spec.split(":")
        count = 1
        if len(parts) not in [1, 2]:
            raise ValidationError("`accelerator` does not match parsing requirements.")
        if len(parts) == 2:
            count = int(parts[1])
        try:
            acc = Accelerator[parts[0]]
        except KeyError as exc:
            raise ValidationError(f"Accelerator {acc_spec} not supported") from exc
        return AcceleratorSpec(accelerator=acc, count=count)


@dataclass
class Resources:
    cpu: str = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    use_gpu: bool = DEFAULT_USE_GPU
    accelerator: AcceleratorSpec = field(default_factory=AcceleratorSpec)

    @staticmethod
    def from_dict(d):
        cpu = d.get("cpu", DEFAULT_CPU)
        validate_cpu_spec(cpu)
        memory = d.get("memory", DEFAULT_MEMORY)
        validate_memory_spec(memory)
        accelerator = AcceleratorSpec.from_str((d.get("accelerator", None)))
        use_gpu = d.get("use_gpu", DEFAULT_USE_GPU)
        if accelerator.accelerator is not None:
            use_gpu = True

        return Resources(
            cpu=cpu,
            memory=memory,
            use_gpu=use_gpu,
            accelerator=accelerator,
        )

    def to_dict(self):
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "use_gpu": self.use_gpu,
            "accelerator": self.accelerator.to_str(),
        }


@dataclass
class Train:
    training_class_filename: str = DEFAULT_TRAINING_CLASS_FILENAME
    training_class_name: str = DEFAULT_TRAINING_CLASS_NAME
    training_module_dir: str = DEFAULT_TRAINING_MODULE_DIR
    variables: Dict = field(default_factory=dict)
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
class ExternalDataItem:
    """A piece of remote data, to be made available to the Truss at serving time.

    Remote data is downloaded and stored under Truss's data directory. Care should be taken
    to avoid conflicts. This will get precedence if there's overlap.
    """

    # Url to download the data from.
    # Currently only files are allowed.
    url: str

    # This should be path relative to data directory. This is where the remote
    # file will be downloaded.
    local_data_path: str

    # The backend used to download files
    backend: str = DEFAULT_BLOB_BACKEND

    # A name can be given to a data item for readability purposes. It's not used
    # in the download process.
    name: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "ExternalDataItem":
        url = d.get("url")
        if url is None or url == "":
            raise ValueError("URL of an external data item cannot be empty")
        local_data_path = d.get("local_data_path")
        if local_data_path is None or local_data_path == "":
            raise ValueError(
                "The `local_data_path` field of an external data item cannot be empty"
            )

        item = ExternalDataItem(
            url=d["url"],
            local_data_path=d["local_data_path"],
            name=d.get("name"),
            backend=d.get("backend", DEFAULT_BLOB_BACKEND),
        )
        return item

    def to_dict(self):
        d = {
            "url": self.url,
            "local_data_path": self.local_data_path,
            "backend": self.backend,
        }
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class ExternalData:
    """[Experimental] External data is data that is not contained in the Truss folder.

    Typically this will be data stored remotely. This data is guaranteed to be made
    available under the data directory of the truss.
    """

    items: List[ExternalDataItem]

    @staticmethod
    def from_list(items: List[Dict[str, str]]) -> "ExternalData":
        return ExternalData([ExternalDataItem.from_dict(item) for item in items])

    def to_list(self) -> List[Dict[str, str]]:
        return [item.to_dict() for item in self.items]


@dataclass
class BaseImage:
    image: str = ""
    python_executable_path: str = ""

    @staticmethod
    def from_dict(d):
        image = d.get("image", "")
        python_executable_path = d.get("python_executable_path", "")
        validate_python_executable_path(python_executable_path)
        return BaseImage(
            image=image,
            python_executable_path=python_executable_path,
        )

    def to_dict(self):
        return {
            "image": self.image,
            "python_executable_path": self.python_executable_path,
        }


@dataclass
class TrussConfig:
    model_framework: ModelFrameworkType = DEFAULT_MODEL_FRAMEWORK_TYPE
    model_type: str = DEFAULT_MODEL_TYPE
    model_name: Optional[str] = None

    model_module_dir: str = DEFAULT_MODEL_MODULE_DIR
    model_class_filename: str = DEFAULT_MODEL_CLASS_FILENAME
    model_class_name: str = DEFAULT_MODEL_CLASS_NAME

    data_dir: str = DEFAULT_DATA_DIRECTORY
    external_data: Optional[ExternalData] = None

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
    description: Optional[str] = None
    bundled_packages_dir: str = DEFAULT_BUNDLED_PACKAGES_DIR
    external_package_dirs: List[str] = field(default_factory=list)
    live_reload: bool = False
    apply_library_patches: bool = True
    # spec_version is a version string
    spec_version: str = DEFAULT_SPEC_VERSION
    train: Train = field(default_factory=Train)
    base_image: Optional[BaseImage] = None

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
            apply_library_patches=d.get("apply_library_patches", True),
            train=Train.from_dict(d.get("train", {})),
            external_data=transform_optional(
                d.get("external_data"), ExternalData.from_list
            ),
            base_image=transform_optional(d.get("base_image"), BaseImage.from_dict),
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
        d = {
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
            "apply_library_patches": self.apply_library_patches,
            "train": self.train.to_dict(),
        }
        if self.external_data is not None:
            d["external_data"] = transform_optional(
                self.external_data, lambda data: data.to_list()
            )
        if self.base_image is not None:
            d["base_image"] = transform_optional(
                self.base_image, lambda data: data.to_dict()
            )
        return d

    def clone(self):
        return TrussConfig.from_dict(self.to_dict())

    def validate(self):
        for secret_name in self.secrets:
            validate_secret_name(secret_name)
