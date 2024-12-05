import sys

if (sys.version_info.major, sys.version_info.minor) <= (3, 8):
    raise RuntimeError(
        "Python version 3.8 or older is not supported for `Truss-Chains`. Please"
        "upgrade to Python 3.9 or newer. You can still use other Truss functionality."
    )
del sys
import pydantic

pydantic_major_version = int(pydantic.VERSION.split(".")[0])
if pydantic_major_version < 2:
    raise RuntimeError(
        f"Pydantic version {pydantic.VERSION} is not supported for Truss-Chains."
        "Please upgrade to pydantic v2. With v1, you can still use all 'classical' "
        "(non-Chains) Truss features."
    )

del pydantic, pydantic_major_version


from truss_chains.definitions import (
    Assets,
    BasetenImage,
    ChainletOptions,
    Compute,
    CustomImage,
    DeployedServiceDescriptor,
    DeploymentContext,
    DockerImage,
    RemoteConfig,
    RemoteErrorDetail,
    RPCOptions,
)
from truss_chains.public_api import (
    ChainletBase,
    depends,
    depends_context,
    mark_entrypoint,
    push,
    run_local,
)

# TODO: make this optional (remove aiohttp, httpx and starlette deps).
from truss_chains.remote_chainlet.stub import StubBase
from truss_chains.utils import make_abs_path_here

__all__ = [
    "Assets",
    "BasetenImage",
    "ChainletBase",
    "ChainletOptions",
    "Compute",
    "CustomImage",
    "DeploymentContext",
    "DockerImage",
    "RPCOptions",
    "RemoteConfig",
    "RemoteErrorDetail",
    "DeployedServiceDescriptor",
    "StubBase",
    "depends",
    "depends_context",
    "make_abs_path_here",
    "mark_entrypoint",
    "push",
    "run_local",
]
