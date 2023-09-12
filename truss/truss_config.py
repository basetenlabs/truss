from dataclasses import _MISSING_TYPE, dataclass, field, fields
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
DEFAULT_PREDICT_CONCURRENCY = 1

DEFAULT_CPU = "1"
DEFAULT_MEMORY = "2Gi"
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
class HuggingFaceModel:
    repo_id: str = ""
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None

    @staticmethod
    def from_dict(d):
        repo_id = d.get("repo_id")
        if repo_id is None or repo_id == "":
            raise ValueError("Repo ID for Hugging Face model cannot be empty.")
        revision = d.get("revision", None)

        allow_patterns = d.get("allow_patterns", None)
        ignore_pattenrs = d.get("ignore_patterns", None)

        return HuggingFaceModel(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_pattenrs,
        )

    def to_dict(self, verbose=False):
        data = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "allow_patterns": self.allow_patterns,
            "ignore_patterns": self.ignore_patterns,
        }

        if not verbose:
            # only show changed values
            data = {k: v for k, v in data.items() if v is not None}

        return data


@dataclass
class HuggingFaceCache:
    models: List[HuggingFaceModel] = field(default_factory=list)

    @staticmethod
    def from_list(items: List[Dict[str, str]]) -> "HuggingFaceCache":
        return HuggingFaceCache([HuggingFaceModel.from_dict(item) for item in items])

    def to_list(self, verbose=False) -> List[Dict[str, str]]:
        return [model.to_dict(verbose=verbose) for model in self.models]


@dataclass
class Runtime:
    predict_concurrency: int = DEFAULT_PREDICT_CONCURRENCY

    @staticmethod
    def from_dict(d):
        predict_concurrency = d.get("predict_concurrency", DEFAULT_PREDICT_CONCURRENCY)

        return Runtime(
            predict_concurrency=predict_concurrency,
        )

    def to_dict(self):
        return {
            "predict_concurrency": self.predict_concurrency,
        }


class ModelServer(Enum):
    TrussServer = "TrussServer"
    TGI = "TGI"
    VLLM = "VLLM"
    TRITON = "TRITON"


@dataclass
class Build:
    model_server: ModelServer = ModelServer.TrussServer
    arguments: Dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d):
        model_server = ModelServer[d.get("model_server", "TrussServer")]
        arguments = d.get("arguments", {})

        return Build(
            model_server=model_server,
            arguments=arguments,
        )

    def to_dict(self):
        return obj_to_dict(self)


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
        return obj_to_dict(self)


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
    """
    `config.yaml` controls Truss config
    Args:
        description (str):
            Describe your model for documentation purposes.
        environment_variables (Dict[str, str]):
            <Warning>
            Do not store secret values directly in environment variables (or anywhere in the
            config file). See the `secrets` arg for information on properly managing secrets.
            </Warning>
            Any environment variables can be provided here as key value pairs and are exposed
            to the environment that the model executes in. Many Python libraries can be
            customized using environment variables, so this field can be quite handy in those
            scenarios.
            ```yaml
            environment_variables:
              ENVIRONMENT: Staging
              DB_URL: https://my_database.example.com/
            ```
        model_metadata (Dict[str, str]):
            Set any additional metadata in this catch-all field. The entire contents of the
            config file are available to the model at runtime, so this is a good place to
            store any custom information that model needs. For example, scikit-learn models
            include a flag here that indicates whether the model supports returning
            probabilities alongside predictions.
            ```yaml
            model_metadata:
              supports_predict_proba: true
            ```
        model_name (str):
            The model's name, for documentation purposes.
        requirements (List[str]):
            List the Python dependencies that the model depends on. The requirements should
            be provided in the
            [pip requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/),
            but as a yaml list.
            We strongly recommend pinning versions in your requirements.
            ```yaml
            requirements:
            - scikit-learn==1.0.2
            - threadpoolctl==3.0.0
            - joblib==1.1.0
            - numpy==1.20.3
            - scipy==1.7.3
            ```
        resources (Dict[str, str]):
            Specify model server runtime resources such as CPU, RAM and GPU.
            ```yaml
            resources:
              cpu: "3"
              memory: 14Gi
              use_gpu: true
              accelerator: A10G
            ```
        secrets (Dict[str, str]):
            <Warning>
            This field can be used to specify the keys for such secrets and dummy default
            values. ***Never store actual secret values in the config***. Dummy default
            values are instructive of what the actual values look like and thus act as
            documentation of the format.
            </Warning>
            A model may depend on certain secret values that can't be bundled with the model
            and need to be bound securely at runtime. For example, a model may need to download
            information from s3 and may need access to AWS credentials for that.
            ```yaml
            secrets:
              hf_access_token: "ACCESS TOKEN"
            ```
        system_packages (List[str]):
            Specify any system packages that you would typically install using `apt` on a Debian operating system.
            ```yaml
            system_packages:
            - ffmpeg
            - libsm6
            - libxext6
            ```
    Returns:
        `config.yaml` file which can be updated
    """

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
    runtime: Runtime = field(default_factory=Runtime)
    build: Build = field(default_factory=Build)
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

    hf_cache: HuggingFaceCache = field(default_factory=HuggingFaceCache)

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
            spec_version=d.get("spec_version", DEFAULT_SPEC_VERSION),
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
            runtime=Runtime.from_dict(d.get("runtime", {})),
            build=Build.from_dict(d.get("build", {})),
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
            hf_cache=transform_optional(
                d.get("hf_cache") or [],
                HuggingFaceCache.from_list,
            ),
        )
        config.validate()
        return config

    @staticmethod
    def from_yaml(yaml_path: Path):
        with yaml_path.open() as yaml_file:
            raw_data = yaml.safe_load(yaml_file) or {}
            return TrussConfig.from_dict(raw_data)

    def write_to_yaml_file(self, path: Path, verbose: bool = True):
        with path.open("w") as config_file:
            yaml.dump(self.to_dict(verbose=verbose), config_file)

    def to_dict(self, verbose: bool = True):
        return obj_to_dict(self, verbose=verbose)

    def clone(self):
        return TrussConfig.from_dict(self.to_dict())

    def validate(self):
        for secret_name in self.secrets:
            validate_secret_name(secret_name)


DATACLASS_TO_REQ_KEYS_MAP = {
    Train: {"variables"},
    Resources: {"accelerator", "cpu", "memory", "use_gpu"},
    Runtime: {"predict_concurrency"},
    Build: {"model_server"},
    TrussConfig: {
        "environment_variables",
        "external_package_dirs",
        "model_metadata",
        "model_name",
        "python_version",
        "requirements",
        "resources",
        "secrets",
        "system_packages",
    },
    BaseImage: {"image", "python_executable_path"},
}


def obj_to_dict(obj, verbose: bool = False):
    """
    This function serializes a given object (usually starting with a TrussConfig) and
    only keeps required keys or ones changed by the user manually. This simplifies the config.yml.
    """
    required_keys = DATACLASS_TO_REQ_KEYS_MAP[type(obj)]
    d = {}
    for f in fields(obj):
        field_name = f.name
        field_default_value = f.default
        field_default_factory = f.default_factory

        field_curr_value = getattr(obj, f.name)

        expected_default_value = None
        if not isinstance(field_default_value, _MISSING_TYPE):
            expected_default_value = field_default_value
        else:
            expected_default_value = field_default_factory()  # type: ignore

        should_add_to_dict = (
            expected_default_value != field_curr_value or field_name in required_keys
        )

        if verbose or should_add_to_dict:
            if isinstance(field_curr_value, tuple(DATACLASS_TO_REQ_KEYS_MAP.keys())):
                d[field_name] = obj_to_dict(field_curr_value, verbose=verbose)
            elif isinstance(field_curr_value, AcceleratorSpec):
                d[field_name] = field_curr_value.to_str()
            elif isinstance(field_curr_value, Enum):
                d[field_name] = field_curr_value.value
            elif isinstance(field_curr_value, ExternalData):
                d["external_data"] = transform_optional(
                    field_curr_value, lambda data: data.to_list()
                )
            elif isinstance(field_curr_value, HuggingFaceCache):
                d["hf_cache"] = transform_optional(
                    field_curr_value, lambda data: data.to_list(verbose=verbose)
                )
            else:
                d[field_name] = field_curr_value

    return d
