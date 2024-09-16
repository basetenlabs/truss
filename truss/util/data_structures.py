from typing import Callable, Optional, TypeVar

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


def transform_optional(x: Optional[X], fn: Callable[[X], Optional[Y]]) -> Optional[Y]:
    if x is None:
        return None

    return fn(x)
