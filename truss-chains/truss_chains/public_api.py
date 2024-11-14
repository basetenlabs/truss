import functools
import pathlib
from typing import ContextManager, Mapping, Optional, Type, Union

from truss_chains import definitions, framework
from truss_chains import remote as chains_remote


def depends_context() -> definitions.DeploymentContext:
    """Sets a "symbolic marker" for injecting a context object at runtime.

    Refer to `the docs <https://docs.baseten.co/chains/getting-started>`_ and this
    `example chainlet <https://github.com/basetenlabs/truss/blob/main/truss-chains/truss_chains/example_chainlet.py>`_
    for more guidance on the ``__init__``-signature of chainlets.

    Warning:
        Despite the type annotation, this does *not* immediately provide a
        context instance. Only when deploying remotely or using ``run_local`` a
        context instance is provided.

    Returns:
        A "symbolic marker" to be used as a default argument in a chainlet's
        initializer.


    """
    # The type error is silenced to because chains framework will at runtime inject
    # a corresponding instance. Nonetheless, we want to use a type annotation here,
    # to facilitate type inference, code-completion and type checking within the code
    # of chainlets that depend on the other chainlet.
    return framework.ContextDependencyMarker()  # type: ignore


def depends(
    chainlet_cls: Type[framework.ChainletT],
    retries: int = 1,
    timeout_sec: int = definitions.DEFAULT_TIMEOUT_SEC,
) -> framework.ChainletT:
    """Sets a "symbolic marker" to indicate to the framework that a chainlet is a
    dependency of another chainlet. The return value of ``depends`` is intended to be
    used as a default argument in a chainlet's ``__init__``-method.
    When deploying a chain remotely, a corresponding stub to the remote is injected in
    its place. In ``run_local`` mode an instance of a local chainlet is injected.

    Refer to `the docs <https://docs.baseten.co/chains/getting-started>`_ and this
    `example chainlet <https://github.com/basetenlabs/truss/blob/main/truss-chains/truss_chains/example_chainlet.py>`_
    for more guidance on how make one chainlet depend on another chainlet.

    Warning:
        Despite the type annotation, this does *not* immediately provide a
        chainlet instance. Only when deploying remotely or using ``run_local`` a
        chainlet instance is provided.


    Args:
        chainlet_cls: The chainlet class of the dependency.
        retries: The number of times to retry the remote chainlet in case of failures
          (e.g. due to transient network issues).
        timeout_sec: Timeout for the HTTP request to this chainlet.

    Returns:
        A "symbolic marker" to be used as a default argument in a chainlet's
        initializer.
    """
    options = definitions.RPCOptions(retries=retries, timeout_sec=timeout_sec)
    # The type error is silenced to because chains framework will at runtime inject
    # a corresponding instance. Nonetheless, we want to use a type annotation here,
    # to facilitate type inference, code-completion and type checking within the code
    # of chainlets that depend on the other chainlet.
    return framework.ChainletDependencyMarker(chainlet_cls, options)  # type: ignore


class ChainletBase(definitions.ABCChainlet):
    """Base class for all chainlets.

    Inheriting from this class adds validations to make sure subclasses adhere to the
    chainlet pattern and facilitates remote chainlet deployment.

    Refer to `the docs <https://docs.baseten.co/chains/getting-started>`_ and this
    `example chainlet <https://github.com/basetenlabs/truss/blob/main/truss-chains/truss_chains/example_chainlet.py>`_
    for more guidance on how to create subclasses.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        framework.validate_and_register_class(cls)  # Errors are collected, not raised!
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
    """Decorator to mark a chainlet as the entrypoint of a chain.

    This decorator can be applied to *one* chainlet in a source file and then the
    CLI push command simplifies because only the file, but not the chainlet class
    in the file, needs to be specified.

    Example usage::

        import truss_chains as chains

        @chains.mark_entrypoint
        class MyChainlet(ChainletBase):
            ...
    """
    return framework.entrypoint(cls)


def push(
    entrypoint: Type[definitions.ABCChainlet],
    chain_name: str,
    publish: bool = True,
    promote: bool = True,
    only_generate_trusses: bool = False,
    remote: Optional[str] = None,
    environment: Optional[str] = None,
) -> chains_remote.BasetenChainService:
    """
    Deploys a chain remotely (with all dependent chainlets).

    Args:
        entrypoint: The chainlet class that serves as the entrypoint to the chain.
        chain_name: The name of the chain.
        publish: Whether to publish the chain as a published deployment (it is a
          draft deployment otherwise)
        promote: Whether to promote the chain to be the production deployment (this
          implies publishing as well).
        only_generate_trusses: Used for debugging purposes. If set to True, only the
          the underlying truss models for the chainlets are generated in
          ``/tmp/.chains_generated``.
        remote: name of a remote config in `.trussrc`. If not provided, it will be
          inquired.
        environment: The name of an environment to promote deployment into.

    Returns:
        A chain service handle to the deployed chain.

    """
    options = definitions.PushOptionsBaseten.create(
        chain_name=chain_name,
        publish=publish,
        promote=promote,
        only_generate_trusses=only_generate_trusses,
        remote=remote,
        environment=environment,
    )
    service = chains_remote.push(entrypoint, options)
    assert isinstance(service, chains_remote.BasetenChainService)  # Per options above.
    return service


def run_local(
    secrets: Optional[Mapping[str, str]] = None,
    data_dir: Optional[Union[pathlib.Path, str]] = None,
    chainlet_to_service: Optional[Mapping[str, definitions.ServiceDescriptor]] = None,
) -> ContextManager[None]:
    """Context manager local debug execution of a chain.

    The arguments only need to be provided if the chainlets explicitly access any the
    corresponding fields of ``DeploymentContext``.

    Args:
        secrets: A dict of secrets keys and values to provide to the chainlets.
        data_dir: Path to a directory with data files.
        chainlet_to_service: A dict of chainlet names to service descriptors.

    Example usage (as trailing main section in a chain file)::

        import os
        import truss_chains as chains


        class HelloWorld(chains.ChainletBase):
            ...


        if __name__ == "__main__":
            with chains.run_local(
                secrets={"some_token": os.environ["SOME_TOKEN"]},
                chainlet_to_service={
                    "SomeChainlet": chains.ServiceDescriptor(
                        name="SomeChainlet",
                        predict_url="https://...",
                        options=chains.RPCOptions(),
                    )
                },
            ):
                hello_world_chain = HelloWorld()
                result = hello_world_chain.run_remote(max_value=5)

            print(result)


    Refer to the `local debugging guide <https://docs.baseten.co/chains/guide#test-a-chain-locally>`_
    for more details.
    """
    data_dir = pathlib.Path(data_dir) if data_dir else None
    return framework.run_local(secrets or {}, data_dir, chainlet_to_service or {})
