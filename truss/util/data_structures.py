from typing import Callable, Dict, Optional, TypeVar

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


def transform_optional(x: Optional[X], fn: Callable[[X], Optional[Y]]) -> Optional[Y]:
    if x is None:
        return None

    return fn(x)


def transform_keys(d: Dict[X, Z], fn: Callable[[X], Y]) -> Dict[Y, Z]:
    return {fn(key): value for key, value in d.items()}
