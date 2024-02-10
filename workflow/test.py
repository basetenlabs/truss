from types import MethodType
from typing import Callable, get_type_hints


class DynamicMethodClass:
    def __init__(self, foo: Callable):
        # Add the function as a method to the instance
        self.foo = MethodType(foo, self)

        # Update type annotations dynamically (for introspection purposes)
        self.__annotations__ = get_type_hints(foo)
        self.__annotations__["foo"] = Callable

        # Alternatively, update the class annotations (affects all instances)
        # DynamicMethodClass.__annotations__.update(get_type_hints(foo))
        # DynamicMethodClass.__annotations__['foo'] = Callable


def example_method(self, x: int, y: int) -> int:
    """Example method that will be added dynamically."""
    return x + y


# Usage
obj = DynamicMethodClass(example_method)


obj.__annotations__
