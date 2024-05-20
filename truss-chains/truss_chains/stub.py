import abc
import functools
import logging
from typing import Optional, Type, TypeVar, final

import httpx
import tenacity
from truss_chains import definitions, utils


class BasetenSession:
    """Helper to invoke predict method on baseten deployments."""

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

    @functools.cached_property
    def _client_sync(self) -> httpx.Client:
        return httpx.Client(
            headers=self._auth_header,
            timeout=self._service_descriptor.options.timeout_sec,
        )

    @functools.cached_property
    def _client_async(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=self._auth_header,
            timeout=self._service_descriptor.options.timeout_sec,
        )

    def predict_sync(self, json_payload):
        retrying = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self._service_descriptor.options.retries),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                if (num := attempt.retry_state.attempt_number) > 1:
                    logging.info(
                        f"Retrying `{self._service_descriptor.name}`, " f"attempt {num}"
                    )
                return utils.handle_response(
                    self._client_sync.post(
                        self._service_descriptor.predict_url, json=json_payload
                    ),
                    self._service_descriptor.name,
                )

    async def predict_async(self, json_payload):
        retrying = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self._service_descriptor.options.retries),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                if (num := attempt.retry_state.attempt_number) > 1:
                    logging.info(
                        f"Retrying `{self._service_descriptor.name}`, " f"attempt {num}"
                    )
                return utils.handle_response(
                    await self._client_async.post(
                        self._service_descriptor.predict_url, json=json_payload
                    ),
                    self._service_descriptor.name,
                )


class StubBase(abc.ABC):
    _remote: BasetenSession

    @final
    def __init__(
        self, service_descriptor: definitions.ServiceDescriptor, api_key: str
    ) -> None:
        self._remote = BasetenSession(service_descriptor, api_key)

    @classmethod
    def from_url(
        cls,
        predict_url: str,
        context: definitions.DeploymentContext,
        options: Optional[definitions.RPCOptions] = None,
        name: Optional[str] = None,
    ):
        name = name or cls.__name__
        options = options or definitions.RPCOptions()
        return cls(
            definitions.ServiceDescriptor(
                name=name, predict_url=predict_url, options=options
            ),
            api_key=context.get_baseten_api_key(),
        )


StubT = TypeVar("StubT", bound=StubBase)


def factory(stub_cls: Type[StubT], context: definitions.DeploymentContext) -> StubT:
    # Assumes the stub_cls-name and the name of the service in `context` match.
    return stub_cls(
        service_descriptor=context.get_service_descriptor(stub_cls.__name__),
        api_key=context.get_baseten_api_key(),
    )
