from typing import Any, ContextManager, Mapping, Optional, Type, final

from slay import definitions, framework, utils


def provide_context() -> Any:
    """Sets a 'symbolic marker' for injecting a Context object at runtime."""
    return framework.ContextProvisionPlaceholder()


def provide(processor_cls: Type[definitions.ABCProcessor]) -> Any:
    """Sets a 'symbolic marker' for injecting a stub or local processor at runtime."""
    # TODO: extend with RPC customization, e.g. timeouts, retries etc.
    return framework.ProcessorProvisionPlaceholder(processor_cls)


class ProcessorBase(definitions.ABCProcessor[definitions.UserConfigT]):

    default_config = definitions.Config()

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
    workflow_name: str,
    publish: bool = True,
    only_generate_trusses: bool = False,
) -> definitions.ServiceDescriptor:
    options = definitions.DeploymentOptionsBaseten(
        workflow_name=workflow_name,
        baseten_url="https://app.baseten.co",
        api_key=utils.get_api_key_from_trussrc(),
        publish=publish,
        only_generate_trusses=only_generate_trusses,
    )
    return framework.deploy_remotely(entrypoint, options)


def run_local(secrets: Optional[Mapping[str, str]] = None) -> ContextManager[None]:
    """Context manager for using in-process instantiations of processor dependencies."""
    return framework.run_local(secrets)
