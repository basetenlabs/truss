import abc
import functools
import logging
from typing import Optional, Type, TypeVar, final

import httpx
from truss_chains import definitions, utils

DEFAULT_TIMEOUT_SEC = 600


class BasetenSession:
    """Helper to invoke predict method on baseten deployments."""

    def __init__(
        self, service_descriptor: definitions.ServiceDescriptor, api_key: str
    ) -> None:
        logging.info(
            f"Creating stub for `{service_descriptor.name}` with predict URL:\n"
            f"\t`{service_descriptor.predict_url}`"
        )
        self._auth_header = {"Authorization": f"Api-Key {api_key}"}
        self._service_descriptor = service_descriptor

    @functools.cached_property
    def _client_sync(self) -> httpx.Client:
        return httpx.Client(headers=self._auth_header, timeout=DEFAULT_TIMEOUT_SEC)

    @functools.cached_property
    def _client_async(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self._auth_header, timeout=DEFAULT_TIMEOUT_SEC)

    def predict_sync(self, json_payload):
        return utils.handle_response(
            self._client_sync.post(
                self._service_descriptor.predict_url, json=json_payload
            )
        )

    async def predict_async(self, json_payload):
        return utils.handle_response(
            await self._client_async.post(
                self._service_descriptor.predict_url, json=json_payload
            )
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
        name: Optional[str] = None,
    ):
        name = name or cls.__name__
        return cls(
            definitions.ServiceDescriptor(name=name, predict_url=predict_url),
            api_key=context.get_baseten_api_key(),
        )


StubT = TypeVar("StubT", bound=StubBase)


def factory(
    stub_cls: Type[StubT], context: definitions.DeploymentContext, chainlet_name: str
) -> StubT:
    return stub_cls(
        service_descriptor=context.get_service_descriptor(chainlet_name),
        api_key=context.get_baseten_api_key(),
    )
