import abc
import asyncio
import logging
import ssl
import threading
import time
from typing import Any, ClassVar, Mapping, Optional, Type, TypeVar, final

import httpx
import tenacity

from truss_chains import definitions, utils


class BasetenSession:
    """Helper to invoke predict method on Baseten deployments."""

    _client_cycle_time_sec: ClassVar[int] = 3600 * 1  # 1 hour.
    _client_limits: ClassVar[httpx.Limits] = httpx.Limits(
        max_connections=1000, max_keepalive_connections=400
    )
    _auth_header: Mapping[str, str]
    _service_descriptor: definitions.ServiceDescriptor
    _cached_sync_client: Optional[tuple[httpx.Client, int]]
    _cached_async_client: Optional[tuple[httpx.AsyncClient, int]]

    def __init__(
        self,
        service_descriptor: definitions.ServiceDescriptor,
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

    def _client_sync(self) -> httpx.Client:
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
        return self._cached_sync_client[0]

    async def _client_async(self) -> httpx.AsyncClient:
        # Check `_client_cycle_needed` before and after locking to avoid
        # needing a lock each time the client is accessed.
        if self._client_cycle_needed(self._cached_async_client):
            async with self._async_lock:
                if self._client_cycle_needed(self._cached_async_client):
                    self._cached_async_client = (
                        httpx.AsyncClient(
                            headers=self._auth_header,
                            timeout=self._service_descriptor.options.timeout_sec,
                            limits=self._client_limits,
                        ),
                        int(time.time()),
                    )
        assert self._cached_async_client is not None
        return self._cached_async_client[0]

    def predict_sync(self, json_payload):
        retrying = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self._service_descriptor.options.retries),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                if (num := attempt.retry_state.attempt_number) > 1:
                    logging.info(f"Retrying `{self.name}`, " f"attempt {num}")
                try:
                    with self._sync_num_requests as num_requests:
                        self._maybe_warn_for_overload(num_requests)
                        resp = self._client_sync().post(
                            self._service_descriptor.predict_url, json=json_payload
                        )
                    return utils.handle_response(resp, self.name)
                # As a special case we invalidate the client in case of certificate
                # errors. This has happened in the past and is a defensive measure.
                except ssl.SSLError:
                    self._cached_sync_client = None
                    raise

    async def predict_async(self, json_payload):
        retrying = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self._service_descriptor.options.retries),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                if (num := attempt.retry_state.attempt_number) > 1:
                    logging.info(f"Retrying `{self.name}`, " f"attempt {num}")
                try:
                    client = await self._client_async()
                    async with self._async_num_requests as num_requests:
                        self._maybe_warn_for_overload(num_requests)
                        resp = await client.post(
                            self._service_descriptor.predict_url, json=json_payload
                        )
                    return utils.handle_response(resp, self.name)
                # As a special case we invalidate the client in case of certificate
                # errors. This has happened in the past and is a defensive measure.
                except ssl.SSLError:
                    self._cached_async_client = None
                    raise


class StubBase(abc.ABC):
    """Base class for stubs that invoke remote chainlets.

    It is used internally for RPCs to dependency chainlets, but it can also be used
    in user-code for wrapping a deployed truss model into the chains framework, e.g.
    like that::

        import pydantic
        import truss_chains as chains

        class WhisperOutput(pydantic.BaseModel):
            ...


        class DeployedWhisper(chains.StubBase):

            async def run_remote(self, audio_b64: str) -> WhisperOutput:
                resp = await self._remote.predict_async(
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

    _remote: BasetenSession

    @final
    def __init__(
        self, service_descriptor: definitions.ServiceDescriptor, api_key: str
    ) -> None:
        """
        Args:
            service_descriptor: Contains the URL and other configuration.
            api_key: A baseten API key to authorize requests.
        """
        self._remote = BasetenSession(service_descriptor, api_key)

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
            definitions.ServiceDescriptor(
                name=cls.__name__, predict_url=predict_url, options=options
            ),
            api_key=context.get_baseten_api_key(),
        )


StubT = TypeVar("StubT", bound=StubBase)


def factory(stub_cls: Type[StubT], context: definitions.DeploymentContext) -> StubT:
    # Assumes the stub_cls-name and the name of the service in ``context` match.
    return stub_cls(
        service_descriptor=context.get_service_descriptor(stub_cls.__name__),
        api_key=context.get_baseten_api_key(),
    )
