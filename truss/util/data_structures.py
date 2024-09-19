from typing import Callable, Optional, TypeVar

X = TypeVar("X")
Y = TypeVar("Y")


def transform_optional(x: Optional[X], fn: Callable[[X], Optional[Y]]) -> Optional[Y]:
    if x is None:
        return None

    return fn(x)
