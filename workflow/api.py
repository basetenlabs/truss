import abc
from typing import Any, Callable, TypeVar

import pydantic

T = TypeVar("T")


class Image:
    ...


class Compute:
    ...


class CustomConfigBase(pydantic.BaseModel):
    ...


class ProcessorBase(abc.ABC):

    # Standardized signature, but can override body.
    def __init__(self, custom_config: CustomConfigBase):
        self._custom_config = custom_config

    @abc.abstractmethod
    def process(self, *args, **kwargs) -> Any:
        ...


def processor(
    name: str, image: Image, compute: Compute, custom_config: CustomConfigBase = None
) -> Callable[[T], T]:
    """
    Just add metadata to obj.
    * Enforce uniqe names.
    * Capture inlined imports.
    * Capture usage of external references.
    * Check that all imports are listed as depenedency.

    """

    def inner(fn_or_cls: T) -> T:
        fn_or_cls.metadata = (name, image, compute)
        return fn_or_cls

    return inner


def model_by_name():
    ...
