# TODO: this file contains too much implementation -> restructure.
import enum
import logging
import pathlib
import traceback
from collections.abc import AsyncIterator
from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import pydantic

from truss.base import custom_types, truss_config

SECRET_DUMMY = "***"
DEFAULT_TIMEOUT_SEC = 600.0
DEFAULT_CONCURRENCY_LIMIT = 300


CpuCountT = Literal["cpu_count"]
CPU_COUNT: CpuCountT = "cpu_count"

_BASETEN_API_SECRET_NAME = "baseten_chain_api_key"

_K = TypeVar("_K", contravariant=True)
_V = TypeVar("_V", covariant=True)


@runtime_checkable
class _MappingNoIter(Protocol[_K, _V]):
    def __getitem__(self, key: _K) -> _V: ...

    def __len__(self) -> int: ...

    def __contains__(self, key: _K) -> bool: ...


### Errors #############################################################################


class ChainsUsageError(TypeError):
    """Raised when user-defined Chainlets do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""


class ChainsRuntimeError(Exception):
    """Raised when components are not used the expected way at runtime."""


class ChainsDeploymentError(Exception):
    """Raised when interaction with a Chain deployment are not possible."""


class GenericRemoteException(Exception):
    """Raised when calling a remote chainlet results in an error and it is not possible
    to re-raise the same exception that was raise remotely in the caller."""


class RemoteErrorDetail(custom_types.SafeModel):
    """When a remote chainlet raises an exception, this pydantic model contains
    information about the error and stack trace and is included in JSON form in the
    error response.
    """

    class StackFrame(custom_types.SafeModel):
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
                filename=self.filename,
                lineno=self.lineno,
                name=self.name,
                line=self.line,
            )

    exception_cls_name: str
    exception_module_name: Optional[str]
    exception_message: str
    user_stack_trace: list[StackFrame]

    def _to_stack_summary(self) -> traceback.StackSummary:
        return traceback.StackSummary.from_list(
            frame.to_frame_summary() for frame in self.user_stack_trace
        )

    def format(self) -> str:
        """Format the error for printing, similar to how Python formats exceptions
        with stack traces."""
        stack = "".join(traceback.format_list(self._to_stack_summary()))
        exc_info = (
            f"\n(Exception class defined in `{self.exception_module_name}`.)"
            if self.exception_module_name
            else ""
        )
        error = (
            f"Chainlet-Traceback (most recent call last):\n"
            f"{stack}{self.exception_cls_name}: {self.exception_message}{exc_info}"
        )
        return error


### Config #############################################################################


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

    def _raise_if_not_exists(self, abs_path: str) -> None:
        path = pathlib.Path(abs_path)
        if not (path.is_file() or (path.is_dir() and any(path.iterdir()))):
            raise MissingDependencyError(
                f"With the file path `{self._original_path}` an absolute path relative "
                f"to the calling module `{self._creating_module}` was created, "
                f"resulting `{self._abs_file_path}` - but no file was found."
            )

    @property
    def abs_path(self) -> str:
        if self._abs_file_path != self._original_path:
            logging.debug(
                f"Using abs path `{self._abs_file_path}` for path specified as "
                f"`{self._original_path}` (in `{self._creating_module}`)."
            )
        abs_path = self._abs_file_path
        self._raise_if_not_exists(abs_path)
        return abs_path


class BasetenImage(enum.Enum):
    """Default images, curated by baseten, for different python versions. If a Chainlet
    uses GPUs, drivers will be included in the image."""

    # Enum values correspond to truss canonical python versions.
    PY39 = "py39"
    PY310 = "py310"
    PY311 = "py311"


class CustomImage(custom_types.SafeModel):
    """Configures the usage of a custom image hosted on dockerhub."""

    image: str
    python_executable_path: Optional[str] = None
    docker_auth: Optional[truss_config.DockerAuthSettings] = None


class DockerImage(custom_types.SafeModelNonSerializable):
    """Configures the docker image in which a remoted chainlet is deployed.

    Note:
        Any paths are relative to the source file where ``DockerImage`` is
        defined and must be created with the helper function ``make_abs_path_here``.
        This allows you for example organize chainlets in different (potentially nested)
        modules and keep their requirement files right next their python source files.

    Args:
        base_image: The base image used by the chainlet. Other dependencies and
          assets are included as additional layers on top of that image. You can choose
          a Baseten default image for a supported python version (e.g.
          ``BasetenImage.PY311``), this will also include GPU drivers if needed, or
          provide a custom image (e.g. ``CustomImage(image="python:3.11-slim")``).
        pip_requirements_file: Path to a file containing pip requirements. The file
          content is naively concatenated with ``pip_requirements``.
        pip_requirements: A list of pip requirements to install.  The items are
          naively concatenated with the content of the ``pip_requirements_file``.
        apt_requirements: A list of apt requirements to install.
        data_dir: Data from this directory is copied into the docker image and
          accessible to the remote chainlet at runtime.
        external_package_dirs: A list of directories containing additional python
          packages outside the chain's workspace dir, e.g. a shared library. This code
          is copied into the docker image and importable at runtime.
        truss_server_version_override: By default, deployed Chainlets use the truss
          server implementation corresponding to the truss version of the user's CLI.
          To use a specific version, e.g. pinning it for exact reproducibility, the
          version can be overridden here. Valid versions correspond to truss releases
          on PyPi: https://pypi.org/project/truss/#history, e.g. `"0.9.80"`.
    """

    base_image: Union[BasetenImage, CustomImage] = BasetenImage.PY311
    pip_requirements_file: Optional[AbsPath] = None
    pip_requirements: list[str] = pydantic.Field(default_factory=list)
    apt_requirements: list[str] = pydantic.Field(default_factory=list)
    data_dir: Optional[AbsPath] = None
    external_package_dirs: Optional[list[AbsPath]] = None
    truss_server_version_override: Optional[str] = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def migrate_fields(cls, values: dict) -> dict:
        if "base_image" in values:
            base_image = values["base_image"]
            if isinstance(base_image, str):
                doc_link = "https://docs.baseten.co/chains-reference/sdk#class-truss-chains-dockerimage"
                raise ChainsUsageError(
                    "`DockerImage.base_image` as string is deprecated. Specify as "
                    f"`BasetenImage` or `CustomImage` (see docs: {doc_link})."
                )
        return values


class ComputeSpec(pydantic.BaseModel):
    """Parsed and validated compute.  See ``Compute`` for more information."""

    # TODO[rcano] add node count
    cpu_count: int = 1
    predict_concurrency: int = 1
    memory: str = "2Gi"
    accelerator: truss_config.AcceleratorSpec = truss_config.AcceleratorSpec()


class Compute:
    """Specifies which compute resources a chainlet has in the *remote* deployment.

    Note:
        Not all combinations can be exactly satisfied by available hardware, in some
        cases more powerful machine types are chosen to make sure requirements are met
        or over-provisioned. Refer to the
        `baseten instance reference <https://docs.baseten.co/deployment/resources>`_.
    """

    # Builder to create ComputeSpec.
    # This extra layer around `ComputeSpec` is needed to parse the accelerator options.

    _spec: ComputeSpec

    def __init__(
        self,
        cpu_count: int = 1,
        memory: str = "2Gi",
        gpu: Union[str, truss_config.Accelerator, None] = None,
        gpu_count: int = 1,
        predict_concurrency: Union[int, CpuCountT] = 1,
    ) -> None:
        """
        Args:
            cpu_count: Minimum number of CPUs to allocate.
            memory: Minimum memory to allocate, e.g. "2Gi" (2 gibibytes).
            gpu: GPU accelerator type, e.g. "A10G", "A100", refer to the
              `truss config <https://docs.baseten.co/reference/config#resources-accelerator>`_
              for more choices.
            gpu_count: Number of GPUs to allocate.
            predict_concurrency: Number of concurrent requests a single replica of a
              deployed chainlet handles.

        Concurrency concepts are explained in `this guide <https://docs.baseten.co/deploy/guides/concurrency#predict-concurrency>`_. # noqa: E501
        It is important to understand the difference between `predict_concurrency` and
        the concurrency target (used for autoscaling, i.e. adding or removing replicas).
        Furthermore, the ``predict_concurrency`` of a single instance is implemented in
        two ways:

        - Via python's ``asyncio``, if ``run_remote`` is an async def. This
          requires that ``run_remote`` yields to the event loop.

        - With a threadpool if it's a synchronous function. This requires
          that the threads don't have significant CPU load (due to the GIL).
        """
        accelerator = truss_config.AcceleratorSpec()
        if gpu:
            accelerator.accelerator = truss_config.Accelerator(gpu)
            accelerator.count = gpu_count
            accelerator = truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator(gpu), count=gpu_count
            )
        if predict_concurrency == CPU_COUNT:
            predict_concurrency_int = cpu_count
        else:
            assert isinstance(predict_concurrency, int)
            predict_concurrency_int = predict_concurrency

        self._spec = ComputeSpec(
            cpu_count=cpu_count,
            memory=memory,
            accelerator=accelerator,
            predict_concurrency=predict_concurrency_int,
        )

    def get_spec(self) -> ComputeSpec:
        return self._spec.model_copy(deep=True)


class AssetSpec(custom_types.SafeModel):
    """Parsed and validated assets. See ``Assets`` for more information."""

    secrets: Mapping[str, str] = pydantic.Field(default_factory=dict)
    cached: list[truss_config.ModelRepo] = pydantic.Field(default_factory=list)
    external_data: list[truss_config.ExternalDataItem] = pydantic.Field(
        default_factory=list
    )


class Assets:
    """Specifies which assets a chainlet can access in the remote deployment.

    For example, model weight caching can be used like this::

        import truss_chains as chains
        from truss.base import truss_config

        mistral_cache = truss_config.ModelRepo(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            allow_patterns=["*.json", "*.safetensors", ".model"]
        )
        chains.Assets(cached=[mistral_cache], ...)

    See `truss caching guide <https://docs.baseten.co/deploy/guides/model-cache#enabling-caching-for-a-model>`_
    for more details on caching.
    """

    # Builder to create asset spec.
    # This extra layer around `AssetSpec` is needed to add secret_keys.
    _spec: AssetSpec

    def __init__(
        self,
        cached: Iterable[truss_config.ModelRepo] = (),
        secret_keys: Iterable[str] = (),
        external_data: Iterable[truss_config.ExternalDataItem] = (),
    ) -> None:
        """
        Args:
            cached: One or more ``truss_config.ModelRepo`` objects.
            secret_keys: Names of secrets stored on baseten, that the
              chainlet should have access to. You can manage secrets on baseten
              `here <https://app.baseten.co/settings/secrets>`_.
            external_data: Data to be downloaded from public URLs and made available
              in the deployment (via ``context.data_dir``). See
              `here <https://docs.baseten.co/reference/config#external-data>`_ for
              more details.
        """
        self._spec = AssetSpec(
            cached=list(cached),
            secrets={k: SECRET_DUMMY for k in secret_keys},
            external_data=list(external_data),
        )

    def get_spec(self) -> AssetSpec:
        """Returns parsed and validated assets."""
        return self._spec.model_copy(deep=True)


try:
    # Was only introduced in 2.5.0
    JsonType = pydantic.JsonValue
except AttributeError:
    JsonType = dict[str, Any]  # type: ignore[misc]


class ChainletOptions(custom_types.SafeModelNonSerializable):
    """
    Args:
        enable_b10_tracing: enables baseten-internal trace data collection. This
          helps baseten engineers better analyze chain performance in case of issues.
          It is independent of a potentially user-configured tracing instrumentation.
          Turning this on, could add performance overhead.
        enable_debug_logs: Sets log level to debug in deployed server.
        env_variables: static environment variables available to the deployed chainlet.
        health_checks: Configures health checks for the chainlet. See `guide <https://docs.baseten.co/truss/guides/custom-health-checks#chains>`_.
        metadata: Arbitrary JSON object to describe chainlet.
        streaming_read_timeout: Amount of time (in seconds) between each streamed chunk before a timeout is triggered.
    """

    enable_b10_tracing: bool = False
    enable_debug_logs: bool = False
    env_variables: Mapping[str, str] = pydantic.Field(default_factory=dict)
    health_checks: truss_config.HealthChecks = truss_config.HealthChecks()
    metadata: Optional[JsonType] = None
    streaming_read_timeout: int = 60


class RemoteConfig(custom_types.SafeModelNonSerializable):
    """Bundles config values needed to deploy a chainlet remotely.

    This is specified as a class variable for each chainlet class, e.g.::

            import truss_chains as chains


            class MyChainlet(chains.ChainletBase):
                remote_config = chains.RemoteConfig(
                    docker_image=chains.DockerImage(
                        pip_requirements=["torch==2.0.1", ...]
                    ),
                    compute=chains.Compute(cpu_count=2, gpu="A10G", ...),
                    assets=chains.Assets(secret_keys=["hf_access_token"], ...),
                )

    """

    docker_image: DockerImage = DockerImage()
    compute: Compute = Compute()
    assets: Assets = Assets()
    name: Optional[str] = None
    options: ChainletOptions = ChainletOptions()

    def get_compute_spec(self) -> ComputeSpec:
        return self.compute.get_spec()

    def get_asset_spec(self) -> AssetSpec:
        return self.assets.get_spec()


class RPCOptions(custom_types.SafeModel):
    """Options to customize RPCs to dependency chainlets.

    Args:
        retries: The number of times to retry the remote chainlet in case of failures
          (e.g. due to transient network issues). For streaming, retries are only made
          if the request fails before streaming any results back. Failures mid-stream
          not retried.
        timeout_sec: Timeout for the HTTP request to this chainlet.
        use_binary: Whether to send data in binary format. This can give a parsing
         speedup and message size reduction (~25%) for numpy arrays. Use
         ``NumpyArrayField`` as a field type on pydantic models for integration and set
         this option to ``True``. For simple text data, there is no significant benefit.
        concurrency_limit: The maximum number of concurrent requests to send to the
          remote chainlet. Excessive requests will be queued and a warning
          will be shown. Try to design your algorithm in a way that spreads requests
          evenly over time so that this the default value can be used.
    """

    retries: int = 1
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    use_binary: bool = False
    concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT


### Interfaces #########################################################################


class WebSocketProtocol(Protocol):
    """Describes subset of starlette/fastAPIs websocket interface that we expose."""

    headers: Mapping[str, str]

    async def close(self, code: int = 1000, reason: Optional[str] = None) -> None: ...

    async def receive_text(self) -> str: ...
    async def receive_bytes(self) -> bytes: ...
    async def receive_json(self) -> Any: ...

    async def send_text(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def send_json(self, data: Any) -> None: ...

    def iter_text(self) -> AsyncIterator[str]: ...
    def iter_bytes(self) -> AsyncIterator[bytes]: ...
    def iter_json(self) -> AsyncIterator[Any]: ...


class EngineBuilderLLMInput(pydantic.BaseModel):
    # TODO: This is mainly a copy from `briton/python/briton/briton/schema.py`.
    #  Find a better way to code share this (potentially in TaT).
    """This class mirrors the `CompletionCreateParamsBase` in the `openai-python` repository.

    However, that class is a Typeddict rather than a pydantic model, so we redefine it here
    to take advantage of pydantic's validation features. In addition, we define helper methods
    to get the formatted prompt, tools to use, and response format to adhere to.

    Unsupported parameters:
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-store
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-metadata
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-frequency_penalty
      - Frequency penalty is not currently passed through to briton
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-logit_bias
      - User provided logit biasing is not implemented
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
      - Returning log probabilities is not implemented
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_logprobs
      - Returning log probabilities is not implemented
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-service_tier
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-user
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
      - Deprecated
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
      - Deprecated
    """

    class Tool(pydantic.BaseModel):
        """An element in the top level `tools` field."""

        class Function(pydantic.BaseModel):
            name: str
            description: Optional[str] = None
            parameters_: Optional[dict[str, Any]] = pydantic.Field(
                None, alias="parameters"
            )
            return_: Optional[dict[str, Any]] = pydantic.Field(None, alias="return")

            @pydantic.model_validator(mode="after")
            def definitions_valid(cls, values):
                if "definitions" in values.parameters and "$defs" in values.parameters:
                    raise ValueError(
                        "Both pydantic v1 and v2 definitions found; please check schema."
                    )
                return values

            @property
            def parameters(self) -> dict[str, Any]:
                if self.parameters_ is None:
                    return {"properties": {}}
                elif "properties" not in self.parameters_:
                    return {"properties": {}, **self.parameters_}
                else:
                    return self.parameters_

            @property
            def parameters_without_definitions(self) -> dict[str, Any]:
                parameters = self.parameters.copy()
                for keyword in ["definitions", "$defs"]:
                    parameters.pop(keyword, None)
                return parameters

            @property
            def definitions(self) -> Optional[tuple[dict[str, Any], str]]:
                for keyword in ["definitions", "$defs"]:
                    if keyword in self.parameters:
                        return self.parameters[keyword], keyword
                return None

            @property
            def json_schema(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "name": {"const": self.name},
                        "parameters": self.parameters_without_definitions,
                    },
                    "required": ["name", "parameters"],
                }

        type: Literal["function"]
        function: Function

    class ToolChoice(pydantic.BaseModel):
        """The top level `tool_choice` field."""

        class FunctionChoice(pydantic.BaseModel):
            name: str

        type: Literal["function"]
        function: FunctionChoice

    class SchemaResponseFormat(pydantic.BaseModel):
        """The top level `response_format` field."""

        class JsonSchema(pydantic.BaseModel):
            """`schema_` holds the actual json schema"""

            schema_: dict[str, Any] = pydantic.Field(..., alias="schema")

        type: Literal["json_schema"]
        json_schema: JsonSchema

    class JsonResponseFormat(pydantic.BaseModel):
        type: Literal["json_object"]

    class TextResponseFormat(pydantic.BaseModel):
        type: Literal["text"]

    class StreamOptions(pydantic.BaseModel):
        """The top level `stream_options` field."""

        include_usage: bool

    class LookaheadDecodingConfig(pydantic.BaseModel):
        window_size: int
        ngram_size: int
        verification_set_size: int

    model: Optional[str] = ""

    messages: Optional[list[dict[str, Any]]] = None
    prompt: Optional[str] = pydantic.Field(None, min_length=1)

    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None

    stream: Optional[bool] = None
    stream_options: Optional[StreamOptions] = None

    seed: Optional[int] = None
    random_seed: Optional[int] = None
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    length_penalty: Optional[float] = None

    # Not part of openai spec but supported by briton
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    runtime_top_p: Optional[float] = None
    top_k: Optional[int] = 50
    runtime_top_k: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    bad_words_: Optional[Union[str, list[str]]] = None
    skip_special_tokens: Optional[list[str]] = None
    response_format: Optional[
        Union[SchemaResponseFormat, JsonResponseFormat, TextResponseFormat]
    ] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Union[Literal["none", "required", "auto"], ToolChoice]] = None
    parallel_tool_calls: Optional[bool] = True
    beam_width: Optional[Literal[1]] = None
    n: Optional[int] = 1
    end_id: Optional[int] = None
    pad_id: Optional[int] = None
    # WiM fields
    margins_prompt: Optional[str] = None
    margins_stop_sequences: Optional[list[str]] = pydantic.Field(
        default_factory=lambda: ["NO#"]
    )
    max_chunk_size: Optional[int] = 4096
    # Lookahead Decoding
    lookahead_decoding_config: Optional[LookaheadDecodingConfig] = None


class DeployedServiceDescriptor(custom_types.SafeModel):
    """Bundles values to establish an RPC session to a dependency chainlet,
    specifically with ``StubBase``."""

    class InternalURL(custom_types.SafeModel):
        gateway_run_remote_url: str  # Includes `https` and endpoint.
        hostname: str  # Does not include `https`.

        def __str__(self) -> str:
            return f"{self.gateway_run_remote_url} (-> {self.hostname})"

    name: str
    display_name: str
    options: RPCOptions
    predict_url: Optional[str] = None
    internal_url: Optional[InternalURL] = pydantic.Field(
        None, description="If provided, takes precedence over `predict_url`."
    )

    @pydantic.model_validator(mode="after")
    def check_at_least_one_url(
        self: "DeployedServiceDescriptor",
    ) -> "DeployedServiceDescriptor":
        if not self.predict_url and not self.internal_url:
            raise ValueError(
                "At least one of 'predict_url' or 'internal_url' must be provided."
            )
        return self


class Environment(custom_types.SafeModel):
    """The environment the chainlet is deployed in.

    Args:
        name: The name of the environment.
    """

    name: str
    # can add more fields here as we add them to dynamic_config configmap


class DeploymentContext(custom_types.SafeModelNonSerializable):
    """Bundles config values and resources needed to instantiate Chainlets.

    The context can optionally be added as a trailing argument in a Chainlet's
    ``__init__`` method and then used to set up the chainlet (e.g. using a secret as
    an access token for downloading model weights).

    Args:
        chainlet_to_service: A mapping from chainlet names to service descriptors.
          This is used to create RPC sessions to dependency chainlets. It contains only
          the chainlet services that are dependencies of the current chainlet.
        secrets: A mapping from secret names to secret values. It contains only the
          secrets that are listed in ``remote_config.assets.secret_keys`` of the
          current chainlet.
        data_dir: The directory where the chainlet can store and access data,
          e.g. for downloading model weights.
        environment: The environment that the chainlet is deployed in.
          None if the chainlet is not associated with an environment.
    """

    chainlet_to_service: Mapping[str, DeployedServiceDescriptor]
    secrets: _MappingNoIter[str, str]
    data_dir: Optional[pathlib.Path] = None
    environment: Optional[Environment] = None

    def get_service_descriptor(self, chainlet_name: str) -> DeployedServiceDescriptor:
        if chainlet_name not in self.chainlet_to_service:
            raise MissingDependencyError(f"{chainlet_name}")
        return self.chainlet_to_service[chainlet_name]

    def get_baseten_api_key(self) -> str:
        if self.secrets is None:
            raise ChainsRuntimeError(
                f"Secrets not set in `{self.__class__.__name__}` object."
            )
        error_msg = (
            "For using chains, it is required to setup a an API key with name "
            f"`{_BASETEN_API_SECRET_NAME}` on Baseten to allow chain Chainlet to "
            "call other Chainlets. For local execution, secrets can be provided "
            "to `run_local`."
        )
        if _BASETEN_API_SECRET_NAME not in self.secrets:
            raise MissingDependencyError(error_msg)

        api_key = self.secrets[_BASETEN_API_SECRET_NAME]
        if api_key == SECRET_DUMMY:
            raise MissingDependencyError(
                f"{error_msg}. Retrieved dummy value of `{api_key}`."
            )
        return api_key
