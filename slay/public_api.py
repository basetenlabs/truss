from typing import Any, ContextManager, Type, final

from slay import definitions, framework


def provide_context() -> Any:
    """Sets a 'symbolic marker' for injecting a Context object at runtime."""
    return framework.ContextProvisionPlaceholder()


def provide(processor_cls: Type[definitions.ABCProcessor]) -> Any:
    """Sets a 'symbolic marker' for injecting a stub or local processor at runtime."""
    return framework.ProcessorProvisionPlaceholder(processor_cls)


class ProcessorBase(definitions.ABCProcessor[definitions.UserConfigT]):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        framework.check_and_register_class(cls)

        original_init = cls.__init__

        def init_with_arg_check(self, *args, **kwargs):
            if args:
                raise definitions.UsageError("Only kwargs are allowed.")
            framework.ensure_args_are_injected(cls, original_init, kwargs)
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


def deploy_remotely(
    entrypoint: Type[definitions.ABCProcessor],
    generate_only: bool = False,
) -> definitions.BasetenRemoteDescriptor:
    return framework.deploy_remotely(entrypoint, generate_only=generate_only)


def run_local() -> ContextManager[None]:
    return framework.run_local()
