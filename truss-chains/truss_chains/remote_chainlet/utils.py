import asyncio
import builtins
import contextlib
import json
import logging
import sys
import textwrap
import threading
import traceback
from typing import (
    Dict,
    Iterator,
    Mapping,
    NoReturn,
    Type,
    TypeVar,
)

import aiohttp
import fastapi
import httpx
import pydantic
from truss.templates.shared import dynamic_config_resolver

from truss_chains import definitions

T = TypeVar("T")


def populate_chainlet_service_predict_urls(
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
) -> Mapping[str, definitions.DeployedServiceDescriptor]:
    chainlet_to_deployed_service: Dict[str, definitions.DeployedServiceDescriptor] = {}

    dynamic_chainlet_config_str = dynamic_config_resolver.get_dynamic_config_value_sync(
        definitions.DYNAMIC_CHAINLET_CONFIG_KEY
    )

    if not dynamic_chainlet_config_str:
        raise definitions.MissingDependencyError(
            f"No '{definitions.DYNAMIC_CHAINLET_CONFIG_KEY}' "
            "found. Cannot override Chainlet configs."
        )

    dynamic_chainlet_config = json.loads(dynamic_chainlet_config_str)

    for (
        chainlet_name,
        service_descriptor,
    ) in chainlet_to_service.items():
        display_name = service_descriptor.display_name

        # NOTE: The Chainlet `display_name` in the Truss CLI
        # corresponds to Chainlet `name` in the backend. As
        # the dynamic Chainlet config is keyed on the backend
        # Chainlet name, we have to look up config values by
        # using the `display_name` in the service descriptor.
        if display_name not in dynamic_chainlet_config:
            raise definitions.MissingDependencyError(
                f"Chainlet '{display_name}' not found in "
                f"'{definitions.DYNAMIC_CHAINLET_CONFIG_KEY}'. "
                f"Dynamic Chainlet config keys: {list(dynamic_chainlet_config)}."
            )

        chainlet_to_deployed_service[chainlet_name] = (
            definitions.DeployedServiceDescriptor(
                display_name=display_name,
                name=service_descriptor.name,
                options=service_descriptor.options,
                predict_url=dynamic_chainlet_config[display_name]["predict_url"],
            )
        )

    return chainlet_to_deployed_service


class AsyncSafeCounter:
    def __init__(self, initial: int = 0) -> None:
        self._counter = initial
        self._lock = asyncio.Lock()

    async def increment(self) -> int:
        async with self._lock:
            self._counter += 1
            return self._counter

    async def decrement(self) -> int:
        async with self._lock:
            self._counter -= 1
            return self._counter

    async def __aenter__(self) -> int:
        return await self.increment()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.decrement()


class ThreadSafeCounter:
    def __init__(self, initial: int = 0) -> None:
        self._counter = initial
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def decrement(self) -> int:
        with self._lock:
            self._counter -= 1
            return self._counter

    def __enter__(self) -> int:
        return self.increment()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.decrement()


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


# Error Propagation Utils. #############################################################


def _handle_exception(exception: Exception, chainlet_name: str) -> NoReturn:
    """Raises `starlette.exceptions.HTTPExceptionn` with `RemoteErrorDetail`."""
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
