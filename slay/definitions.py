# TODO: this file contains too much implementation -> restructure.
import abc
import inspect
import logging
import os
import traceback
from types import GenericAlias
from typing import Any, ClassVar, Generic, Mapping, Optional, Type, TypeVar

import pydantic
from pydantic import generics

UserConfigT = TypeVar("UserConfigT", bound=Optional[pydantic.BaseModel])

BASETEN_API_SECRET_NAME = "baseten_workflow_api_key"
SECRET_DUMMY = "***"
TRUSS_CONFIG_SLAY_KEY = "slay_metadata"

ENDPOINT_METHOD_NAME = "run"  # Referring to processor method name exposed as endpoint.
# Below arg names must correspond to `definitions.ABCProcessor`.
CONTEXT_ARG_NAME = "context"  # Referring to processors `__init__` signature.
SELF_ARG_NAME = "self"

GENERATED_CODE_DIR = ".slay_generated"
PREDICT_ENDPOINT_NAME = "/predict"
PROCESSOR_MODULE = "processor"


class APIDefinitionError(TypeError):
    """Raised when user-defined processors do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""


class UsageError(Exception):
    """Raised when components are not used the expected way at runtime."""


class AbsPath:
    _abs_file_path: str
    _creating_module: str
    _original_path: str

    def __init__(
        self, abs_file_path: str, creating_module: str, original_path: str
    ) -> None:
        self._abs_file_path = abs_file_path
        self._creating_module = creating_module
        self._original_path = original_path

    def raise_if_not_exists(self) -> None:
        if not os.path.isfile(self._abs_file_path):
            raise MissingDependencyError(
                f"With the file path `{self._original_path}` an absolute path relative "
                f"to the calling module `{self._creating_module}` was created, "
                f"resulting `{self._abs_file_path}` - but no file was found."
            )

    @property
    def abs_path(self) -> str:
        return self._abs_file_path


def make_abs_path_here(file_path: str) -> AbsPath:
    """Helper to specify file paths relative to the *immediately calling* module.

    E.g. in you have a project structure like this

    root/
        workflow.py
        common_requirements.text
        sub_package/
            processor.py
            processor_requirements.txt

    Not in `root/sub_package/processor.py` you can point to the requirements
    file like this:

    ```
    shared = RelativePathToHere("../common_requirements.text")
    specific = RelativePathToHere("processor_requirements.text")
    ```

    Caveat: this helper uses the directory of the immediately calling module as an
    absolute reference point for resolving the file location.
    Therefore, you MUST NOT wrap the instantiation of `RelativePathToHere` into a
    function (e.g. applying decorators) or use dynamic code execution.

    Ok:
    ```
    def foo(path: AbsPath):
        abs_path = path.abs_path


    foo(make_abs_path_here("blabla"))
    ```

    Not Ok:
    ```
    def foo(path: str):
        badbadbad = make_abs_path_here(path).abs_path

    foo("blabla"))
    ```
    """
    # TODO: the absolute path resolution below uses the calling module as a
    #   reference point. This would not work if users wrap this call in a function
    #   - we hope the naming makes clear that this should not be done.
    caller_frame = inspect.stack()[1]
    module_path = caller_frame.filename
    if not os.path.isabs(file_path):
        module_dir = os.path.dirname(os.path.abspath(module_path))
        abs_file_path = os.path.normpath(os.path.join(module_dir, file_path))
        logging.info(f"Inferring absolute path for `{file_path}` as `{abs_file_path}`.")
    else:
        abs_file_path = file_path

    return AbsPath(abs_file_path, module_path, file_path)


class ImageSpec(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # TODO: this is not stable yet and might change or refer back to truss.
    base_image: str = "python:3.11-slim"
    pip_requirements_file: Optional[AbsPath] = None
    pip_requirements: list[str] = []
    apt_requirements: list[str] = []


class Image:
    """Builder to create image spec."""

    _spec: ImageSpec

    def __init__(self) -> None:
        self._spec = ImageSpec()

    def pip_requirements_file(self, file_path: AbsPath) -> "Image":
        self._spec.pip_requirements_file = file_path
        return self

    def pip_requirements(self, requirements: list[str]) -> "Image":
        self._spec.pip_requirements = requirements
        return self

    def apt_requirements(self, requirements: list[str]) -> "Image":
        self._spec.apt_requirements = requirements
        return self

    def get_spec(self) -> ImageSpec:
        return self._spec.copy(deep=True)


class ComputeSpec(pydantic.BaseModel):
    # TODO: this is not stable yet and might change or refer back to truss.
    cpu: str = "1"
    memory: str = "2Gi"
    gpu: Optional[str] = None


class Compute:
    """Builder to create compute spec."""

    _spec: ComputeSpec

    def __init__(self) -> None:
        self._spec = ComputeSpec()

    def cpu(self, cpu: int) -> "Compute":
        self._spec.cpu = str(cpu)
        return self

    def memory(self, memory: str) -> "Compute":
        self._spec.memory = memory
        return self

    def gpu(self, kind: str, count: int = 1) -> "Compute":
        self._spec.gpu = f"{kind}:{count}"
        return self

    def get_spec(self) -> ComputeSpec:
        return self._spec.copy(deep=True)


class AssetSpec(pydantic.BaseModel):
    # TODO: this is not stable yet and might change or refer back to truss.
    secrets: dict[str, str] = {}
    cached: list[Any] = []


class Assets:
    """Builder to create asset spec."""

    _spec: AssetSpec

    def __init__(self) -> None:
        self._spec = AssetSpec()

    def add_secret(self, key: str) -> "Assets":
        # Actual value is provided in deployment.
        self._spec.secrets[key] = SECRET_DUMMY
        return self

    def cached(self, value: list[Any]) -> "Assets":
        self._spec.cached = value
        return self

    def get_spec(self) -> AssetSpec:
        return self._spec.copy(deep=True)


class Config(generics.GenericModel, Generic[UserConfigT]):
    """Bundles config values needed to deploy a processor."""

    class Config:
        arbitrary_types_allowed = True

    name: Optional[str] = None
    image: Image = Image()
    compute: Compute = Compute()
    assets: Assets = Assets()
    user_config: UserConfigT = pydantic.Field(default=None)

    def get_image_spec(self) -> ImageSpec:
        return self.image.get_spec()

    def get_compute_spec(self) -> ComputeSpec:
        return self.compute.get_spec()

    def get_asset_spec(self) -> AssetSpec:
        return self.assets.get_spec()


class ServiceDescriptor(pydantic.BaseModel):
    name: str
    predict_url: str


class Context(generics.GenericModel, Generic[UserConfigT]):
    """Bundles config values needed to instantiate a processor in deployment."""

    class Config:
        arbitrary_types_allowed = True

    user_config: UserConfigT = pydantic.Field(default=None)
    stub_cls_to_service: Mapping[str, ServiceDescriptor] = {}
    # secrets: Optional[secrets_resolver.Secrets] = None
    # TODO: above type results in `truss.server.shared.secrets_resolver.Secrets`
    # due to the templating, at runtime the object passed will be from
    # `shared.secrets_resolver` and give pydantic validation error.
    secrets: Optional[Any] = None

    def get_service_descriptor(self, stub_cls_name: str) -> ServiceDescriptor:
        if stub_cls_name not in self.stub_cls_to_service:
            raise MissingDependencyError(f"{stub_cls_name}")
        return self.stub_cls_to_service[stub_cls_name]

    def get_baseten_api_key(self) -> str:
        if self.secrets is None:
            raise UsageError(f"Secrets not set in `{self.__class__.__name__}` object.")
        error_msg = (
            "For using workflows, it is required to setup a an API key with name "
            f"`{BASETEN_API_SECRET_NAME}` on baseten to allow workflow processor to "
            "call other processors. For local execution, secrets can be provided "
            "to `run_local`."
        )
        if BASETEN_API_SECRET_NAME not in self.secrets:
            raise MissingDependencyError(error_msg)

        api_key = self.secrets[BASETEN_API_SECRET_NAME]
        if api_key == SECRET_DUMMY:
            raise MissingDependencyError(
                f"{error_msg}. Retrieved dummy value of `{api_key}`."
            )
        return api_key


class TrussMetadata(generics.GenericModel, Generic[UserConfigT]):
    """Plugin for the truss config (in config["model_metadata"]["slay_metadata"])."""

    class Config:
        arbitrary_types_allowed = True

    user_config: UserConfigT = pydantic.Field(default=None)
    stub_cls_to_service: Mapping[str, ServiceDescriptor] = {}


class ABCProcessor(Generic[UserConfigT], abc.ABC):
    default_config: ClassVar[Config]
    _init_is_patched: ClassVar[bool] = False
    _context: Context[UserConfigT]

    @abc.abstractmethod
    def __init__(self, context: Context[UserConfigT]) -> None:
        ...

    # Cannot add this abstract method to API, because we want to allow arbitrary
    # arg/kwarg names and specifying any function signature here would give type errors
    # @abc.abstractmethod
    # def predict(self, *args, **kwargs) -> Any: ...

    @property
    @abc.abstractmethod
    def user_config(self) -> UserConfigT:
        ...


class TypeDescriptor(pydantic.BaseModel):
    """For describing I/O types of processors."""

    # TODO: Supporting pydantic types.

    raw: Any  # The raw type annotation object (could be a type or GenericAlias).

    def as_src_str(self) -> str:
        if isinstance(self.raw, type):
            return self.raw.__name__
        else:
            return str(self.raw)

    @property
    def is_pydantic(self) -> bool:
        return (
            isinstance(self.raw, type)
            and not isinstance(self.raw, GenericAlias)
            and issubclass(self.raw, pydantic.BaseModel)
        )


class EndpointAPIDescriptor(pydantic.BaseModel):
    name: str = ENDPOINT_METHOD_NAME
    input_names_and_types: list[tuple[str, TypeDescriptor]]
    output_types: list[TypeDescriptor]
    is_async: bool
    is_generator: bool


class ProcessorAPIDescriptor(pydantic.BaseModel):
    processor_cls: Type[ABCProcessor]
    src_path: str
    dependencies: Mapping[str, Type[ABCProcessor]]
    endpoint: EndpointAPIDescriptor
    user_config_type: TypeDescriptor

    def __hash__(self) -> int:
        return hash(self.processor_cls)

    @property
    def cls_name(self) -> str:
        return self.processor_cls.__name__


class DeploymentOptions(pydantic.BaseModel):
    workflow_name: str
    only_generate_trusses: bool = False


class DeploymentOptionsBaseten(DeploymentOptions):
    baseten_url: str
    api_key: str
    publish: bool


class DeploymentOptionsLocalDocker(DeploymentOptions):
    # Local docker-to-docker requests don't need auth, but we need to set a
    # value different from `SECRET_DUMMY` to not trigger the check that the secret
    # is unset. Additionally, if local docker containers make calls to models deployed
    # on baseten, a real API key must be provided (i.e. the default must be overridden).
    baseten_workflow_api_key: str = "docker_dummy_key"


class StackFrame(pydantic.BaseModel):
    filename: str
    lineno: Optional[int]
    name: str
    line: Optional[str]

    @classmethod
    def from_frame_summary(cls, frame: traceback.FrameSummary):
        return cls(
            filename=frame.filename,
            lineno=frame.lineno,
            name=frame.name,
            line=frame.line,
        )

    def to_frame_summary(self) -> traceback.FrameSummary:
        return traceback.FrameSummary(
            filename=self.filename, lineno=self.lineno, name=self.name, line=self.line
        )


class RemoteErrorDetail(pydantic.BaseModel):
    remote_name: str
    exception_class_name: str
    exception_module_name: Optional[str]
    exception_message: str
    user_stack_trace: list[StackFrame]

    def to_stack_summary(self) -> traceback.StackSummary:
        return traceback.StackSummary.from_list(
            frame.to_frame_summary() for frame in self.user_stack_trace
        )

    def format(self) -> str:
        stack = "".join(traceback.format_list(self.to_stack_summary()))
        exc_info = (
            f"\n(Exception class defined in `{self.exception_module_name}`.)"
            if self.exception_module_name
            else ""
        )
        error = (
            f"{RemoteErrorDetail.__name__} in `{self.remote_name}`\n"
            f"Traceback (most recent call last):\n"
            f"{stack}{self.exception_class_name}: {self.exception_message}{exc_info}"
        )
        return error


class GenericRemoteException(Exception):
    ...
