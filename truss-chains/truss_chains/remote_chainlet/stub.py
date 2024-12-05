import abc
import asyncio
import contextlib
import contextvars
import json
import logging
import threading
import time
from typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    final,
    overload,
)

import aiohttp
import httpx
import pydantic
import starlette.requests
import tenacity
from truss.templates.shared import serialization

from truss_chains import definitions
from truss_chains.remote_chainlet import utils

DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 400


_RetryPolicyT = TypeVar("_RetryPolicyT", tenacity.AsyncRetrying, tenacity.Retrying)
_InputT = TypeVar("_InputT", pydantic.BaseModel, Any)  # Any signifies "JSON".
_OutputT = TypeVar("_OutputT", bound=pydantic.BaseModel)


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
            f"Creating BasetenSession (HTTP) for `{service_descriptor.name}`.\n"
            f"\tTarget: `{service_descriptor.predict_url}`\n"
            f"\t`{service_descriptor.options}`."
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
        logging.info(f"Retrying `{self.name}`, attempt {retry_state.attempt_number}")

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
            utils.response_raise_errors(response, self.name)
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
                    await utils.async_response_raise_errors(response, self.name)
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
                await utils.async_response_raise_errors(response, self.name)
                return response.content.iter_any()

        return await retry(_rpc)


StubT = TypeVar("StubT", bound=StubBase)


def factory(stub_cls: Type[StubT], context: definitions.DeploymentContext) -> StubT:
    # Assumes the stub_cls-name and the name of the service in ``context` match.
    return stub_cls(
        service_descriptor=context.get_service_descriptor(stub_cls.__name__),
        api_key=context.get_baseten_api_key(),
    )
