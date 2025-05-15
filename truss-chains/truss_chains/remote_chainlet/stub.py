import abc
import asyncio
import contextlib
import json
import logging
import threading
import time
from typing import (
    TYPE_CHECKING,
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

import httpx
import pydantic
import tenacity

from truss.templates.shared import serialization
from truss_chains import private_types, public_types
from truss_chains import utils as chains_utils
from truss_chains.remote_chainlet import utils

if TYPE_CHECKING:
    import aiohttp


_RetryPolicyT = TypeVar("_RetryPolicyT", tenacity.AsyncRetrying, tenacity.Retrying)
InputT = TypeVar("InputT", pydantic.BaseModel, Any)  # Any signifies "JSON".
OutputModelT = TypeVar("OutputModelT", bound=pydantic.BaseModel)


async def _safe_close(session: "aiohttp.ClientSession", timeout_sec: float) -> None:
    try:
        await asyncio.wait_for(session.close(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logging.info("Timeout while closing cycled-out aiohttp session.")
    except Exception as e:
        logging.info(f"Unexpected error while closing cycled-out aiohttp session: {e}")


class BasetenSession:
    """Provides configured HTTP clients, retries, queueing etc."""

    _client_cycle_time_sec: ClassVar[int] = 3600  # 1 hour.
    _target_url: str
    _headers: Mapping[str, str]
    _service_descriptor: public_types.DeployedServiceDescriptor
    _client_limits: httpx.Limits
    _cached_sync_client: Optional[tuple[httpx.Client, int]]
    _cached_async_client: Optional[tuple["aiohttp.ClientSession", int]]
    _sync_lock: threading.Lock
    _async_lock: asyncio.Lock
    _sync_semaphore_wrapper: utils.ThreadSemaphoreWrapper
    _async_semaphore_wrapper: utils.AsyncSemaphoreWrapper
    _close_tasks: list[asyncio.Task[None]]

    def __init__(
        self, service_descriptor: public_types.DeployedServiceDescriptor, api_key: str
    ) -> None:
        headers = {"Authorization": f"Api-Key {api_key}"}
        # If `internal_url` is present, it takes precedence.
        if service_descriptor.internal_url:
            target_msg = str(service_descriptor.internal_url)
            headers["Host"] = service_descriptor.internal_url.hostname
            target_url = service_descriptor.internal_url.gateway_run_remote_url
        elif service_descriptor.predict_url:
            target_msg = service_descriptor.predict_url
            target_url = service_descriptor.predict_url
        else:
            assert False, (
                "Per validation of `DeployedServiceDescriptor` either `predict_url` "
                f"or `internal_url` must be present. Got {service_descriptor}"
            )

        logging.info(
            f"Creating BasetenSession (HTTP) for `{service_descriptor.name}`.\n"
            f"\tTarget: `{target_msg}`\n\t`{service_descriptor.options}`."
        )
        self._target_url = target_url
        self._headers = headers
        self._service_descriptor = service_descriptor
        self._client_limits = httpx.Limits(
            max_connections=service_descriptor.options.concurrency_limit,
            max_keepalive_connections=50,
        )
        self._cached_sync_client = None
        self._cached_async_client = None
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._sync_semaphore_wrapper = utils.ThreadSemaphoreWrapper(
            service_descriptor.options.concurrency_limit, service_descriptor.name
        )
        self._async_semaphore_wrapper = utils.AsyncSemaphoreWrapper(
            service_descriptor.options.concurrency_limit, service_descriptor.name
        )
        self._close_tasks = []

    @property
    def name(self) -> str:
        return self._service_descriptor.name

    async def shut_down(self) -> None:
        # TODO: integrate this with the uvicorn server.
        await asyncio.gather(*self._close_tasks)

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
                            headers=self._headers,
                            timeout=self._service_descriptor.options.timeout_sec,
                            limits=self._client_limits,
                        ),
                        int(time.time()),
                    )
        assert self._cached_sync_client is not None
        client = self._cached_sync_client[0]

        with self._sync_semaphore_wrapper():
            yield client

    @contextlib.asynccontextmanager
    async def _client_async(self) -> AsyncIterator["aiohttp.ClientSession"]:
        try:
            import aiohttp
        except ImportError:
            raise chains_utils.make_optional_import_error("aiohttp")

        # Check `_client_cycle_needed` before and after locking to avoid
        # needing a lock each time the client is accessed.
        if self._client_cycle_needed(self._cached_async_client):
            async with self._async_lock:
                if self._client_cycle_needed(self._cached_async_client):
                    if self._cached_async_client is not None:
                        # Close with same timeout as connections, but add some buffer.
                        self._close_tasks.append(
                            asyncio.create_task(
                                _safe_close(
                                    self._cached_async_client[0],
                                    self._service_descriptor.options.timeout_sec * 1.1,
                                )
                            )
                        )
                    limit = self._client_limits.max_connections
                    assert limit is not None
                    connector = aiohttp.TCPConnector(limit=limit)
                    self._cached_async_client = (
                        aiohttp.ClientSession(
                            headers=self._headers,
                            connector=connector,
                            timeout=aiohttp.ClientTimeout(
                                total=self._service_descriptor.options.timeout_sec
                            ),
                        ),
                        int(time.time()),
                    )
        assert self._cached_async_client is not None
        client = self._cached_async_client[0]

        async with self._async_semaphore_wrapper():
            yield client


class StubBase(BasetenSession, abc.ABC):
    """Base class for stubs that invoke remote chainlets.

    Extends ``BasetenSession`` with methods for data serialization, de-serialization
    and invoking other endpoints.

    It is used internally for RPCs to dependency chainlets, but it can also be used
    in user-code for wrapping a deployed truss model into the Chains framework. It
    flexibly supports JSON and pydantic inputs and output. Example usage::

        import pydantic
        import truss_chains as chains


        class WhisperOutput(pydantic.BaseModel):
            ...


        class DeployedWhisper(chains.StubBase):
            # Input JSON, output JSON.
            async def run_remote(self, audio_b64: str) -> Any:
                return await self.predict_async(
                    inputs={"audio": audio_b64})
                # resp == {"text": ..., "language": ...}

            # OR Input JSON, output pydantic model.
            async def run_remote(self, audio_b64: str) -> WhisperOutput:
                return await self.predict_async(
                    inputs={"audio": audio_b64}, output_model=WhisperOutput)

            # OR Input and output are pydantic models.
            async def run_remote(self, data: WhisperInput) -> WhisperOutput:
                return await self.predict_async(data, output_model=WhisperOutput)


        class MyChainlet(chains.ChainletBase):

            def __init__(self, ..., context=chains.depends_context()):
                ...
                self._whisper = DeployedWhisper.from_url(
                    WHISPER_URL,
                    context,
                    options=chains.RPCOptions(retries=3),
                )

            async def run_remote(self, ...):
               await self._whisper.run_remote(...)
    """

    @final
    def __init__(
        self, service_descriptor: public_types.DeployedServiceDescriptor, api_key: str
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
        context_or_api_key: Union[public_types.DeploymentContext, str],
        options: Optional[public_types.RPCOptions] = None,
    ):
        """Factory method, convenient to be used in chainlet's ``__init__``-method.

        Args:
            predict_url: URL to predict endpoint of another chain / truss model.
            context_or_api_key: Deployment context object, obtained in the
               chainlet's ``__init__`` or Baseten API key.
            options: RPC options, e.g. retries.
        """
        options = options or public_types.RPCOptions()
        if isinstance(context_or_api_key, str):
            api_key = context_or_api_key
        else:
            api_key = context_or_api_key.get_baseten_api_key()
        return cls(
            service_descriptor=public_types.DeployedServiceDescriptor(
                name=cls.__name__,
                display_name=cls.__name__,
                predict_url=predict_url,
                options=options,
            ),
            api_key=api_key,
        )

    def _make_request_params(
        self, inputs: InputT, for_httpx: bool = False
    ) -> Mapping[str, Any]:
        kwargs: Dict[str, Any] = {}
        headers = {}
        if trace_parent := utils.get_trace_parent():
            headers[private_types.OTEL_TRACE_PARENT_HEADER_KEY] = trace_parent

        if isinstance(inputs, pydantic.BaseModel):
            if self._service_descriptor.options.use_binary:
                data_dict = inputs.model_dump(mode="python")
                data_key = "content" if for_httpx else "data"
                kwargs[data_key] = serialization.truss_msgpack_serialize(data_dict)
                headers["Content-Type"] = "application/octet-stream"
            else:
                data_key = "content" if for_httpx else "data"
                kwargs[data_key] = inputs.model_dump_json()
                headers["Content-Type"] = "application/json"
        else:  # inputs is JSON dict.
            if self._service_descriptor.options.use_binary:
                data_key = "content" if for_httpx else "data"
                kwargs[data_key] = serialization.truss_msgpack_serialize(inputs)
                headers["Content-Type"] = "application/octet-stream"
            else:
                kwargs["json"] = inputs
                headers["Content-Type"] = "application/json"

        kwargs["headers"] = headers
        return kwargs

    def _response_to_pydantic(
        self, response: bytes, output_model: Type[OutputModelT]
    ) -> OutputModelT:
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
        self, inputs: InputT, output_model: Type[OutputModelT]
    ) -> OutputModelT:
        """Returns a validated pydantic model. Inputs can be pydantic or JSON dict."""

    @overload
    def predict_sync(self, inputs: InputT, output_model: None = None) -> Any:
        """Returns a raw JSON dict. Inputs can be pydantic or JSON dict."""

    def predict_sync(
        self, inputs: InputT, output_model: Optional[Type[OutputModelT]] = None
    ) -> Union[OutputModelT, Any]:
        retry = self._make_retry_policy(tenacity.Retrying)
        params = self._make_request_params(inputs, for_httpx=True)

        def _rpc() -> bytes:
            client: httpx.Client
            with self._client_sync() as client:
                response = client.post(self._target_url, **params)
            utils.response_raise_errors(response, self.name)
            return response.content

        try:
            response_bytes = retry(_rpc)
        except httpx.ReadTimeout:
            msg = (
                f"Timeout calling remote Chainlet `{self.name}` "
                f"({self._service_descriptor.options.timeout_sec} seconds limit)."
            )
            logging.warning(msg)
            raise TimeoutError(msg) from None  # Prune error stack trace (TMI).

        if output_model:
            return self._response_to_pydantic(response_bytes, output_model)
        return self._response_to_json(response_bytes)

    @overload
    async def predict_async(
        self, inputs: InputT, output_model: Type[OutputModelT]
    ) -> OutputModelT:
        """Returns a validated pydantic model. Inputs can be pydantic or JSON dict."""

    @overload
    async def predict_async(self, inputs: InputT, output_model: None = None) -> Any:
        """Returns a validated pydantic model. Inputs can be pydantic or JSON dict."""

    async def predict_async(
        self, inputs: InputT, output_model: Optional[Type[OutputModelT]] = None
    ) -> Union[OutputModelT, Any]:
        retry = self._make_retry_policy(tenacity.AsyncRetrying)
        params = self._make_request_params(inputs)

        async def _rpc() -> bytes:
            client: "aiohttp.ClientSession"
            async with self._client_async() as client:
                async with client.post(self._target_url, **params) as response:
                    await utils.async_response_raise_errors(response, self.name)
                    return await response.read()

        try:
            response_bytes: bytes = await retry(_rpc)
        except asyncio.TimeoutError:
            msg = (
                f"Timeout calling remote Chainlet `{self.name}` "
                f"({self._service_descriptor.options.timeout_sec} seconds limit)."
            )
            logging.warning(msg)
            raise TimeoutError(msg) from None  # Prune error stack trace (TMI).

        if output_model:
            return self._response_to_pydantic(response_bytes, output_model)
        return self._response_to_json(response_bytes)

    async def predict_async_stream(self, inputs: InputT) -> AsyncIterator[bytes]:
        retry = self._make_retry_policy(tenacity.AsyncRetrying)
        params = self._make_request_params(inputs)

        async def _rpc() -> AsyncIterator[bytes]:
            client: "aiohttp.ClientSession"
            async with self._client_async() as client:
                response = await client.post(self._target_url, **params)
                await utils.async_response_raise_errors(response, self.name)
                return response.content.iter_any()

        try:
            return await retry(_rpc)
        except asyncio.TimeoutError:
            msg = (
                f"Timeout calling remote Chainlet `{self.name}` "
                f"({self._service_descriptor.options.timeout_sec} seconds limit)."
            )
            logging.warning(msg)
            raise TimeoutError(msg) from None  # Prune error stack trace (TMI).


StubT = TypeVar("StubT", bound=StubBase)


def factory(stub_cls: Type[StubT], context: public_types.DeploymentContext) -> StubT:
    # Assumes the stub_cls-name and the name of the service in ``context` match.
    return stub_cls(
        service_descriptor=context.get_service_descriptor(stub_cls.__name__),
        api_key=context.get_baseten_api_key(),
    )
