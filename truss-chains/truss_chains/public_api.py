import inspect
import pathlib
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
    Mapping,
    Optional,
    Type,
    Union,
    overload,
)

from truss_chains import framework, private_types, public_types
from truss_chains.deployment import deployment_client

if TYPE_CHECKING:
    from rich import progress


def depends_context() -> public_types.DeploymentContext:
    """Sets a "symbolic marker" for injecting a context object at runtime.

    Refer to `the docs <https://docs.baseten.co/chains/getting-started>`_ and this
    `example chainlet <https://github.com/basetenlabs/truss/blob/main/truss-chains/truss_chains/reference_code/reference_chainlet.py>`_
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
    timeout_sec: float = public_types._DEFAULT_TIMEOUT_SEC,
    use_binary: bool = False,
) -> framework.ChainletT:
    """Sets a "symbolic marker" to indicate to the framework that a chainlet is a
    dependency of another chainlet. The return value of ``depends`` is intended to be
    used as a default argument in a chainlet's ``__init__``-method.
    When deploying a chain remotely, a corresponding stub to the remote is injected in
    its place. In ``run_local`` mode an instance of a local chainlet is injected.

    Refer to `the docs <https://docs.baseten.co/chains/getting-started>`_ and this
    `example chainlet <https://github.com/basetenlabs/truss/blob/main/truss-chains/truss_chains/reference_code/reference_chainlet.py>`_
    for more guidance on how make one chainlet depend on another chainlet.

    Warning:
        Despite the type annotation, this does *not* immediately provide a
        chainlet instance. Only when deploying remotely or using ``run_local`` a
        chainlet instance is provided.


    Args:
        chainlet_cls: The chainlet class of the dependency.
        retries: The number of times to retry the remote chainlet in case of failures
          (e.g. due to transient network issues). For streaming, retries are only made
          if the request fails before streaming any results back. Failures mid-stream
          not retried.
        timeout_sec: Timeout for the HTTP request to this chainlet.
        use_binary: Whether to send data in binary format. This can give a parsing
         speedup and message size reduction (~25%) for numpy arrays. Use
         ``NumpyArrayField`` as a field type on pydantic models for integration and set
         this option to ``True``. For simple text data, there is no significant benefit.

    Returns:
        A "symbolic marker" to be used as a default argument in a chainlet's
        initializer.
    """
    options = public_types.RPCOptions(
        retries=retries, timeout_sec=timeout_sec, use_binary=use_binary
    )
    # The type error is silenced to because chains framework will at runtime inject
    # a corresponding instance. Nonetheless, we want to use a type annotation here,
    # to facilitate type inference, code-completion and type checking within the code
    # of chainlets that depend on the other chainlet.
    return framework.ChainletDependencyMarker(chainlet_cls, options)  # type: ignore


@overload
def mark_entrypoint(
    cls_or_chain_name: Type[framework.ChainletT],
) -> Type[framework.ChainletT]: ...


@overload
def mark_entrypoint(
    cls_or_chain_name: str,
) -> Callable[[Type[framework.ChainletT]], Type[framework.ChainletT]]: ...


def mark_entrypoint(
    cls_or_chain_name: Union[str, Type[framework.ChainletT]],
) -> Union[
    Type[framework.ChainletT],
    Callable[[Type[framework.ChainletT]], Type[framework.ChainletT]],
]:
    """Decorator to mark a chainlet as the entrypoint of a chain.

    This decorator can be applied to *one* chainlet in a source file and then the
    CLI push command simplifies: only the file, not the class within, must be specified.

    Optionally a display name for the Chain (not the Chainlet) can be set (effectively
    giving a custom default value for the `--name` arg of the CLI push command).

    Example usage::

        import truss_chains as chains

        @chains.mark_entrypoint
        class MyChainlet(ChainletBase):
            ...

        # OR with custom Chain name.
        @chains.mark_entrypoint("My Chain Name")
        class MyChainlet(ChainletBase):
            ...
    """
    return framework.entrypoint(cls_or_chain_name)


def push(
    entrypoint: Type[framework.ChainletT],
    chain_name: str,
    publish: bool = True,
    promote: bool = True,
    only_generate_trusses: bool = False,
    remote: str = "baseten",
    environment: Optional[str] = None,
    progress_bar: Optional[Type["progress.Progress"]] = None,
    include_git_info: bool = False,
) -> deployment_client.BasetenChainService:
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
        progress_bar: Optional `rich.progress.Progress` if output is desired.
        include_git_info: Whether to attach git versioning info (sha, branch, tag) to
          deployments made from within a git repo. If set to True in `.trussrc`, it
          will always be attached.

    Returns:
        A chain service handle to the deployed chain.

    """
    options = private_types.PushOptionsBaseten.create(
        chain_name=chain_name,
        publish=publish,
        promote=promote,
        only_generate_trusses=only_generate_trusses,
        remote=remote,
        environment=environment,
        include_git_info=include_git_info,
        working_dir=pathlib.Path(inspect.getfile(entrypoint)).parent,
    )
    service = deployment_client.push(entrypoint, options, progress_bar=progress_bar)
    assert isinstance(
        service, deployment_client.BasetenChainService
    )  # Per options above.
    return service


def run_local(
    secrets: Optional[Mapping[str, str]] = None,
    data_dir: Optional[Union[pathlib.Path, str]] = None,
    chainlet_to_service: Optional[
        Mapping[str, public_types.DeployedServiceDescriptor]
    ] = None,
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
                    "SomeChainlet": chains.DeployedServiceDescriptor(
                        name="SomeChainlet",
                        display_name="SomeChainlet",
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
