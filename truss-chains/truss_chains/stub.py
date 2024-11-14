import abc
import asyncio
import builtins
import contextlib
import contextvars
import json
import logging
import sys
import textwrap
import threading
import time
import traceback
from typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Dict,
    Iterator,
    Mapping,
    NoReturn,
    Optional,
    Type,
    TypeVar,
    Union,
    final,
    overload,
)

import aiohttp
import fastapi
import httpx
import pydantic
import starlette.requests
import tenacity
from truss.templates.shared import serialization

from truss_chains import definitions, utils

DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 400


_RetryPolicyT = TypeVar("_RetryPolicyT", tenacity.AsyncRetrying, tenacity.Retrying)
_InputT = TypeVar("_InputT", pydantic.BaseModel, Any)  # Any signifies "JSON".
_OutputT = TypeVar("_OutputT", bound=pydantic.BaseModel)


# Error Propagation Utils. #############################################################


def _handle_exception(exception: Exception, chainlet_name: str) -> NoReturn:
    """Raises `fastapi.HTTPException` with `RemoteErrorDetail` as detail."""
    if hasattr(exception, "__module__"):
        exception_module_name = exception.__module__
    else:
        exception_module_name = None

    error_stack = traceback.extract_tb(exception.__traceback__)
    # Exclude the error handling functions from the stack trace.
    exclude_frames = {
        exception_to_http_error.__name__,
        response_raise_errors.__name__,
        async_response_raise_errors.__name__,
    }
    final_tb = [frame for frame in error_stack if frame.name not in exclude_frames]
    stack = list(
        [definitions.StackFrame.from_frame_summary(frame) for frame in final_tb]
    )
    error = definitions.RemoteErrorDetail(
        remote_name=chainlet_name,
        exception_cls_name=exception.__class__.__name__,
        exception_module_name=exception_module_name,
        exception_message=str(exception),
        user_stack_trace=stack,
    )
    raise fastapi.HTTPException(
        status_code=500, detail=error.model_dump()
    ) from exception


@contextlib.contextmanager
def exception_to_http_error(chainlet_name: str) -> Iterator[None]:
    # TODO: move chainlet name from here to caller side.
    try:
        yield
    except Exception as e:
        _handle_exception(e, chainlet_name)


def _resolve_exception_class(
    error: definitions.RemoteErrorDetail,
) -> Type[Exception]:
    """Tries to find the exception class in builtins or imported libs,
    falls back to `definitions.GenericRemoteError` if not found."""
    exception_cls = None
    if error.exception_module_name is None:
        exception_cls = getattr(builtins, error.exception_cls_name, None)
    else:
        if mod := sys.modules.get(error.exception_module_name):
            exception_cls = getattr(mod, error.exception_cls_name, None)

    if exception_cls is None:
        logging.warning(
            f"Could not resolve exception with name `{error.exception_cls_name}` "
            f"and module `{error.exception_module_name}` - fall back to "
            f"`{definitions.GenericRemoteException.__name__}`."
        )
        exception_cls = definitions.GenericRemoteException

    if issubclass(exception_cls, pydantic.ValidationError):
        # Cannot re-raise naively.
        # https://github.com/pydantic/pydantic/issues/6734.
        exception_cls = definitions.GenericRemoteException

    return exception_cls


def _handle_response_error(response_json: dict, remote_name: str):
    try:
        error_json = response_json["error"]
    except KeyError as e:
        logging.error(f"response_json: {response_json}")
        raise ValueError(
            "Could not get `error` field from JSON from error response"
        ) from e
    try:
        error = definitions.RemoteErrorDetail.model_validate(error_json)
    except pydantic.ValidationError as e:
        if isinstance(error_json, str):
            msg = f"Remote error occurred in `{remote_name}`: '{error_json}'"
            raise definitions.GenericRemoteException(msg) from None
        raise ValueError(
            "Could not parse error. Error details are expected to be either a "
            "plain string (old truss models) or a serialized "
            f"`definitions.RemoteErrorDetail.__name__`, got:\n{repr(error_json)}"
        ) from e
    exception_cls = _resolve_exception_class(error)
    msg = (
        f"(showing remote errors, root message at the bottom)\n"
        f"--> Preceding Remote Cause:\n"
        f"{textwrap.indent(error.format(), '    ')}"
    )
    raise exception_cls(msg)


def response_raise_errors(response: httpx.Response, remote_name: str) -> None:
    """In case of error, raise it.

    If the response error contains `RemoteErrorDetail`, it tries to re-raise
    the same exception that was raised remotely and falls back to
    `GenericRemoteException` if the exception class could not be resolved.

    Exception messages are chained to trace back to the root cause, i.e. the first
    Chainlet that raised an exception. E.g. the message might look like this:

    ```
    RemoteChainletError in "Chain"
    Traceback (most recent call last):
      File "/app/model/Chainlet.py", line 112, in predict
        result = await self._chainlet.run(
      File "/app/model/Chainlet.py", line 79, in run
        value += self._text_to_num.run(part)
      File "/packages/remote_stubs.py", line 21, in run
        json_result = self.predict_sync(json_args)
      File "/packages/truss_chains/stub.py", line 37, in predict_sync
        return utils.handle_response(
    ValueError: (showing remote errors, root message at the bottom)
    --> Preceding Remote Cause:
        RemoteChainletError in "TextToNum"
        Traceback (most recent call last):
          File "/app/model/Chainlet.py", line 113, in predict
            result = self._chainlet.run(data=payload["data"])
          File "/app/model/Chainlet.py", line 54, in run
            generated_text = self._replicator.run(data)
          File "/packages/remote_stubs.py", line 7, in run
            json_result = self.predict_sync(json_args)
          File "/packages/truss_chains/stub.py", line 37, in predict_sync
            return utils.handle_response(
        ValueError: (showing remote errors, root message at the bottom)
        --> Preceding Remote Cause:
            RemoteChainletError in "TextReplicator"
            Traceback (most recent call last):
              File "/app/model/Chainlet.py", line 112, in predict
                result = self._chainlet.run(data=payload["data"])
              File "/app/model/Chainlet.py", line 36, in run
                raise ValueError(f"This input is too long: {len(data)}.")
            ValueError: This input is too long: 100.

    ```
    """
    if response.is_error:
        try:
            response_json = response.json()
        except Exception as e:
            raise ValueError(
                "Could not get JSON from error response. Status: "
                f"`{response.status_code}`."
            ) from e
        _handle_response_error(response_json=response_json, remote_name=remote_name)


async def async_response_raise_errors(
    response: aiohttp.ClientResponse, remote_name: str
) -> None:
    """Async version of `async_response_raise_errors`."""
    if response.status >= 400:
        try:
            response_json = await response.json()
        except Exception as e:
            raise ValueError(
                "Could not get JSON from error response. Status: "
                f"`{response.status}`."
            ) from e
        _handle_response_error(response_json=response_json, remote_name=remote_name)


########################################################################################


_trace_parent_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_parent"
)


@contextlib.contextmanager
def trace_parent(request: starlette.requests.Request) -> Iterator[None]:
    token = _trace_parent_context.set(
        request.headers.get(definitions.OTEL_TRACE_PARENT_HEADER_KEY, "")
    )
    try:
        yield
    finally:
        _trace_parent_context.reset(token)


def pydantic_set_field_dict(obj: pydantic.BaseModel) -> dict[str, pydantic.BaseModel]:
    """Like `BaseModel.model_dump(exclude_unset=True), but only top-level.

    This is used to get kwargs for invoking a function, while dropping fields for which
    there is no value explicitly set in the pydantic model. A field is considered unset
    if the key was not present in the incoming JSON request (from which the model was
    parsed/initialized) and the pydantic model has a default value, such as `None`.

    By dropping these unset fields, the default values from the function definition
    will be used instead. This behavior ensures correct handling of arguments where
    the function has a default, such as in the case of `run_remote`. If the model has
    an optional field defaulting to `None`, this approach differentiates between
    the user explicitly passing a value of `None` and the field being unset in the
    request.

    """
    return {name: getattr(obj, name) for name in obj.model_fields_set}


class BasetenSession:
    """Provides configured HTTP clients, retries rate limit warning etc."""

    _client_cycle_time_sec: ClassVar[int] = 3600 * 1  # 1 hour.
    _client_limits: ClassVar[httpx.Limits] = httpx.Limits(
        max_connections=DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections=DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
    )
    _auth_header: Mapping[str, str]
    _service_descriptor: definitions.DeployedServiceDescriptor
    _cached_sync_client: Optional[tuple[httpx.Client, int]]
    _cached_async_client: Optional[tuple[aiohttp.ClientSession, int]]

    def __init__(
        self,
        service_descriptor: definitions.DeployedServiceDescriptor,
        api_key: str,
    ) -> None:
        logging.info(
            f"Creating BasetenSession (HTTP) for `{service_descriptor.name}` "
            f"({service_descriptor.options.retries} retries) with predict URL:\n"
            f"    `{service_descriptor.predict_url}`"
        )
        self._auth_header = {"Authorization": f"Api-Key {api_key}"}
        self._service_descriptor = service_descriptor
        self._cached_sync_client = None
        self._cached_async_client = None
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._sync_num_requests = utils.ThreadSafeCounter()
        self._async_num_requests = utils.AsyncSafeCounter()

    @property
    def name(self) -> str:
        return self._service_descriptor.name

    def _maybe_warn_for_overload(self, num_requests: int) -> None:
        if self._client_limits.max_connections is None:
            return
        if num_requests > self._client_limits.max_connections * 0.8:
            logging.warning(
                f"High number of concurrently outgoing HTTP connections: "
                f"`{num_requests}`. Close to or above connection limit of "
                f"`{self._client_limits.max_connections}`. To avoid overload and "
                f"timeouts, use more replicas/autoscaling for this chainlet."
            )

    def _client_cycle_needed(self, cached_client: Optional[tuple[Any, int]]) -> bool:
        return (
            not cached_client
            or (int(time.time()) - cached_client[1]) > self._client_cycle_time_sec
        )

    def _log_retry(self, retry_state: tenacity.RetryCallState) -> None:
        attempt_number = retry_state.attempt_number
        if attempt_number > 1:
            logging.info(f"Retrying `{self.name}`, attempt {attempt_number}")

    def _make_retry_policy(self, retrying: Type[_RetryPolicyT]) -> _RetryPolicyT:
        return retrying(
            stop=tenacity.stop_after_attempt(self._service_descriptor.options.retries),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=self._log_retry,
        )

    @contextlib.contextmanager
    def _client_sync(self) -> Iterator[httpx.Client]:
        # Check `_client_cycle_needed` before and after locking to avoid
        # needing a lock each time the client is accessed.
        if self._client_cycle_needed(self._cached_sync_client):
            with self._sync_lock:
                if self._client_cycle_needed(self._cached_sync_client):
                    self._cached_sync_client = (
                        httpx.Client(
                            headers=self._auth_header,
                            timeout=self._service_descriptor.options.timeout_sec,
                            limits=self._client_limits,
                        ),
                        int(time.time()),
                    )
        assert self._cached_sync_client is not None
        client = self._cached_sync_client[0]

        with self._sync_num_requests as num_requests:
            self._maybe_warn_for_overload(num_requests)
            yield client

    @contextlib.asynccontextmanager
    async def _client_async(self) -> AsyncIterator[aiohttp.ClientSession]:
        # Check `_client_cycle_needed` before and after locking to avoid
        # needing a lock each time the client is accessed.
        if self._client_cycle_needed(self._cached_async_client):
            async with self._async_lock:
                if self._client_cycle_needed(self._cached_async_client):
                    connector = aiohttp.TCPConnector(
                        limit=DEFAULT_MAX_CONNECTIONS,
                    )
                    self._cached_async_client = (
                        aiohttp.ClientSession(
                            headers=self._auth_header,
                            connector=connector,
                            timeout=aiohttp.ClientTimeout(
                                total=self._service_descriptor.options.timeout_sec
                            ),
                        ),
                        int(time.time()),
                    )
        assert self._cached_async_client is not None
        client = self._cached_async_client[0]

        async with self._async_num_requests as num_requests:
            self._maybe_warn_for_overload(num_requests)
            yield client


class StubBase(BasetenSession, abc.ABC):
    """Base class for stubs that invoke remote chainlets.

    Extends ``BasetenSession`` with methods for data serialization, de-serialization
    and invoking other endpoints.

    It is used internally for RPCs to dependency chainlets, but it can also be used
    in user-code for wrapping a deployed truss model into the chains framework, e.g.
    like that::

        import pydantic
        import truss_chains as chains

        class WhisperOutput(pydantic.BaseModel):
            ...


        class DeployedWhisper(chains.StubBase):

            async def run_remote(self, audio_b64: str) -> WhisperOutput:
                resp = await self.predict_async(
                    json_payload={"audio": audio_b64})
                return WhisperOutput(text=resp["text"], language=resp["language"])


        class MyChainlet(chains.ChainletBase):

            def __init__(self, ..., context=chains.depends_context()):
                ...
                self._whisper = DeployedWhisper.from_url(
                    WHISPER_URL,
                    context,
                    options=chains.RPCOptions(retries=3),
                )

    """

    @final
    def __init__(
        self,
        service_descriptor: definitions.DeployedServiceDescriptor,
        api_key: str,
    ) -> None:
        """
        Args:
            service_descriptor: Contains the URL and other configuration.
            api_key: A baseten API key to authorize requests.
        """
        super().__init__(service_descriptor, api_key)

    @classmethod
    def from_url(
        cls,
        predict_url: str,
        context: definitions.DeploymentContext,
        options: Optional[definitions.RPCOptions] = None,
    ):
        """Factory method, convenient to be used in chainlet's ``__init__``-method.

        Args:
            predict_url: URL to predict endpoint of another chain / truss model.
            context: Deployment context object, obtained in the chainlet's ``__init__``.
            options: RPC options, e.g. retries.
        """
        options = options or definitions.RPCOptions()
        return cls(
            service_descriptor=definitions.DeployedServiceDescriptor(
                name=cls.__name__,
                display_name=cls.__name__,
                predict_url=predict_url,
                options=options,
            ),
            api_key=context.get_baseten_api_key(),
        )

    def _make_request_params(
        self, inputs: _InputT, for_httpx: bool = False
    ) -> Mapping[str, Any]:
        kwargs: Dict[str, Any] = {}
        headers = {
            definitions.OTEL_TRACE_PARENT_HEADER_KEY: _trace_parent_context.get()
        }
        if isinstance(inputs, pydantic.BaseModel):
            if self._service_descriptor.options.use_binary:
                data_dict = inputs.model_dump(mode="python")
                kwargs["data"] = serialization.truss_msgpack_serialize(data_dict)
                headers["Content-Type"] = "application/octet-stream"
            else:
                data_key = "content" if for_httpx else "data"
                kwargs[data_key] = inputs.model_dump_json()
                headers["Content-Type"] = "application/json"
        else:  # inputs is JSON dict.
            if self._service_descriptor.options.use_binary:
                kwargs["data"] = serialization.truss_msgpack_serialize(inputs)
                headers["Content-Type"] = "application/octet-stream"
            else:
                kwargs["json"] = inputs
                headers["Content-Type"] = "application/json"

        kwargs["headers"] = headers
        return kwargs

    def _response_to_pydantic(
        self, response: bytes, output_model: Type[_OutputT]
    ) -> _OutputT:
        if self._service_descriptor.options.use_binary:
            data_dict = serialization.truss_msgpack_deserialize(response)
            return output_model.model_validate(data_dict)
        return output_model.model_validate_json(response)

    def _response_to_json(self, response: bytes) -> Any:
        if self._service_descriptor.options.use_binary:
            return serialization.truss_msgpack_deserialize(response)
        return json.loads(response)

    @overload
    def predict_sync(
        self, inputs: _InputT, output_model: Type[_OutputT]
    ) -> _OutputT: ...

    @overload  # Returns JSON
    def predict_sync(self, inputs: _InputT, output_model: None = None) -> Any: ...

    def predict_sync(
        self, inputs: _InputT, output_model: Optional[Type[_OutputT]] = None
    ) -> Union[_OutputT, Any]:
        retry = self._make_retry_policy(tenacity.Retrying)
        params = self._make_request_params(inputs, for_httpx=True)

        def _rpc() -> bytes:
            client: httpx.Client
            with self._client_sync() as client:
                response = client.post(self._service_descriptor.predict_url, **params)
            response_raise_errors(response, self.name)
            return response.content

        response_bytes = retry(_rpc)
        if output_model:
            return self._response_to_pydantic(response_bytes, output_model)
        return self._response_to_json(response_bytes)

    @overload
    async def predict_async(
        self, inputs: _InputT, output_model: Type[_OutputT]
    ) -> _OutputT: ...

    @overload  # Returns JSON.
    async def predict_async(
        self, inputs: _InputT, output_model: None = None
    ) -> Any: ...

    async def predict_async(
        self, inputs: _InputT, output_model: Optional[Type[_OutputT]] = None
    ) -> Union[_OutputT, Any]:
        retry = self._make_retry_policy(tenacity.AsyncRetrying)
        params = self._make_request_params(inputs)

        async def _rpc() -> bytes:
            client: aiohttp.ClientSession
            async with self._client_async() as client:
                async with client.post(
                    self._service_descriptor.predict_url, **params
                ) as response:
                    await async_response_raise_errors(response, self.name)
                    return await response.read()

        response_bytes: bytes = await retry(_rpc)
        if output_model:
            return self._response_to_pydantic(response_bytes, output_model)
        return self._response_to_json(response_bytes)

    async def predict_async_stream(self, inputs: _InputT) -> AsyncIterator[bytes]:
        retry = self._make_retry_policy(tenacity.AsyncRetrying)
        params = self._make_request_params(inputs)

        async def _rpc() -> AsyncIterator[bytes]:
            client: aiohttp.ClientSession
            async with self._client_async() as client:
                response = await client.post(
                    self._service_descriptor.predict_url, **params
                )
                await async_response_raise_errors(response, self.name)
                return response.content.iter_any()

        return await retry(_rpc)


StubT = TypeVar("StubT", bound=StubBase)


def factory(stub_cls: Type[StubT], context: definitions.DeploymentContext) -> StubT:
    # Assumes the stub_cls-name and the name of the service in ``context` match.
    return stub_cls(
        service_descriptor=context.get_service_descriptor(stub_cls.__name__),
        api_key=context.get_baseten_api_key(),
    )
