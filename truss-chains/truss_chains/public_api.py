import pathlib
from typing import Any, ContextManager, Mapping, Optional, Type, Union, final

from truss_chains import definitions, deploy, framework


def provide_context() -> Any:
    """Sets a 'symbolic marker' for injecting a Context object at runtime."""
    return framework.ContextProvisionMarker()


def provide(chainlet_cls: Type[definitions.ABCChainlet], retries: int = 1) -> Any:
    """Sets a 'symbolic marker' for injecting a stub or local Chainlet at runtime."""
    options = definitions.RPCOptions(retries=retries)
    return framework.ChainletProvisionMarker(chainlet_cls, options)


class ChainletBase(definitions.ABCChainlet[definitions.UserConfigT]):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        framework.check_and_register_class(cls)

        original_init = cls.__init__

        def init_with_arg_check(self, *args, **kwargs):
            if args:
                raise definitions.ChainsRuntimeError("Only kwargs are allowed.")
            framework.ensure_args_are_injected(cls, original_init, kwargs)
            original_init(self, *args, **kwargs)

        cls.__init__ = init_with_arg_check  # type: ignore[method-assign]

    def __init__(
        self,
        context: definitions.DeploymentContext[
            definitions.UserConfigT
        ] = provide_context(),
    ) -> None:
        self._context = context

    @final
    @property
    def user_config(self) -> definitions.UserConfigT:
        return self._context.user_config


def deploy_remotely(
    entrypoint: Type[definitions.ABCChainlet],
    chain_name: str,
    publish: bool = True,
    promote: bool = True,
    only_generate_trusses: bool = False,
) -> deploy.ChainService:
    options = definitions.DeploymentOptionsBaseten.create(
        chain_name=chain_name,
        publish=publish,
        promote=promote,
        only_generate_trusses=only_generate_trusses,
    )
    return deploy.deploy_remotely(entrypoint, options)


def run_local(
    secrets: Optional[Mapping[str, str]] = None,
    data_dir: Optional[Union[pathlib.Path, str]] = None,
) -> ContextManager[None]:
    """Context manager for using in-process instantiations of Chainlet dependencies."""
    data_dir = pathlib.Path(data_dir) if data_dir else None
    return framework.run_local(secrets, data_dir)
