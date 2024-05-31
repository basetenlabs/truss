import functools
import pathlib
from typing import ContextManager, Mapping, Optional, Type, Union

from truss_chains import definitions, deploy, framework


def depends_context() -> definitions.DeploymentContext:
    """Sets a 'symbolic marker' for injecting a Context object at runtime.

    WARNING: Despite the type annotation, this does *not* immediately provide a
    context instance.
    Only when deploying remotely or using `run_local` a context instance is provided.
    """
    # The type error is silenced to because chains framework will at runtime inject
    # a corresponding instance. Nonetheless, we want to use a type annotation here,
    # to facilitate type inference, code-completion and type checking within the code
    # of chainlets that depend on the other chainlet.
    return framework.ContextDependencyMarker()  # type: ignore


def depends(
    chainlet_cls: Type[framework.ChainletT], retries: int = 1
) -> framework.ChainletT:
    """Sets a 'symbolic marker' for injecting a stub or local Chainlet at runtime.

    WARNING: Despite the type annotation, this does *not* immediately provide a
    chainlet instance.
    Only when deploying remotely or using `run_local` a chainlet instance is provided.
    """
    options = definitions.RPCOptions(retries=retries)
    # The type error is silenced to because chains framework will at runtime inject
    # a corresponding instance. Nonetheless, we want to use a type annotation here,
    # to facilitate type inference, code-completion and type checking within the code
    # of chainlets that depend on the other chainlet.
    return framework.ChainletDependencyMarker(chainlet_cls, options)  # type: ignore


class ChainletBase(definitions.ABCChainlet):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        framework.check_and_register_class(cls)
        # For default init (from `object`) we don't need to check anything.
        if cls.has_custom_init():
            original_init = cls.__init__

            @functools.wraps(original_init)
            def __init_with_arg_check__(self, *args, **kwargs):
                if args:
                    raise definitions.ChainsRuntimeError("Only kwargs are allowed.")
                framework.ensure_args_are_injected(cls, original_init, kwargs)
                original_init(self, *args, **kwargs)

            cls.__init__ = __init_with_arg_check__  # type: ignore[method-assign]


def mark_entrypoint(cls: Type[framework.ChainletT]) -> Type[framework.ChainletT]:
    return framework.entrypoint(cls)


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
    chainlet_to_service: Optional[Mapping[str, definitions.ServiceDescriptor]] = None,
) -> ContextManager[None]:
    """Context manager for using in-process instantiations of Chainlet dependencies."""
    data_dir = pathlib.Path(data_dir) if data_dir else None
    return framework.run_local(secrets, data_dir, chainlet_to_service)
