from typing import Any, Iterable, Type, final

from slay import definitions, framework


def provide_context() -> Any:
    return framework.ContextProvisionPlaceholder()


def provide(processor_cls: Type[definitions.ABCProcessor]) -> Any:
    return framework.ProcessorProvisionPlaceholder(processor_cls)


class ProcessorBase(definitions.ABCProcessor[definitions.UserConfigT]):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        framework.check_and_register_class(cls)

        original_init = cls.__init__

        def init_with_arg_check(self, *args, **kwargs):
            if args:
                raise definitions.UsageError("Only kwargs are allowed.")
            framework.check_init_args(cls, original_init, kwargs)
            original_init(self, *args, **kwargs)

        cls.__init__ = init_with_arg_check  # type: ignore[method-assign]

    def __init__(
        self, context: definitions.Context[definitions.UserConfigT] = provide_context()
    ) -> None:
        self._context = context

    @final
    @property
    def user_config(self) -> definitions.UserConfigT:
        return self._context.user_config


def deploy_remotely(processors: Iterable[Type[definitions.ABCProcessor]]) -> None:
    return framework.deploy_remotely(processors)


def run_local() -> Any:
    return framework.run_local()
