from typing import Any, Iterable, Type

from slay import definitions, framework


def provide_config() -> Any:
    return framework.ConfigProvisionPlaceholder()


def provide(processor_cls: Type[definitions.ABCProcessor]) -> Any:
    return framework.ProcessorProvisionPlaceholder(processor_cls)


class ProcessorBase(definitions.ABCProcessor[definitions.UserConfigT]):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        framework.check_and_register_class(cls)

        original_init = cls.__init__

        def modified_init(self, *args, **kwargs):
            assert not args
            framework.check_init_args(cls, original_init, kwargs)
            original_init(self, *args, **kwargs)

        cls.__init__ = modified_init  # type: ignore[method-assign]

    def __init__(
        self, config: definitions.Config[definitions.UserConfigT] = provide_config()
    ) -> None:
        config.user_config
        self._config = config

    @property
    def user_config(self) -> definitions.UserConfigT:
        return self._config.user_config


def deploy_remotely(processors: Iterable[Type[definitions.ABCProcessor]]) -> None:
    return framework.deploy_remotely(processors)


def run_local() -> Any:
    return framework.run_local()
