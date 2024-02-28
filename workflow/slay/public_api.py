from typing import Any, Generic, Iterable, Type

from . import framework, types


def provide_config() -> Any:
    return framework.ConfigProvisionPlaceholder()


def provide(processor_cls: Type[types.ABCProcessor]) -> Any:
    return framework.ProcessorProvisionPlaceholder(processor_cls)


class BaseProcessor(types.ABCProcessor[types.UserConfigT], Generic[types.UserConfigT]):
    def __init__(
        self, config: types.Config[types.UserConfigT] = provide_config()
    ) -> None:
        self._config = config

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        framework.check_and_register_class(cls)

        original_init = cls.__init__

        def modified_init(self, *args, **kwargs):
            assert not args
            framework.check_init_args(cls, kwargs)
            original_init(self, *args, **kwargs)

        cls.__init__ = modified_init  # type: ignore[method-assign]


def deploy_remotely(processors: Iterable[Type[types.ABCProcessor]]) -> None:
    return framework.deploy_remotely(processors)


def run_local() -> Any:
    return framework.run_local()
