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
    Compute,
    CustomImage,
    DeploymentContext,
    DockerImage,
    RemoteConfig,
    RemoteErrorDetail,
    RPCOptions,
    ServiceDescriptor,
)
from truss_chains.public_api import (
    ChainletBase,
    depends,
    depends_context,
    deploy_remotely,
    mark_entrypoint,
    run_local,
)
from truss_chains.stub import StubBase
from truss_chains.utils import make_abs_path_here

__all__ = [
    "Assets",
    "BasetenImage",
    "ChainletBase",
    "Compute",
    "CustomImage",
    "DeploymentContext",
    "DockerImage",
    "RPCOptions",
    "RemoteConfig",
    "RemoteErrorDetail",
    "ServiceDescriptor",
    "StubBase",
    "depends",
    "depends_context",
    "deploy_remotely",
    "make_abs_path_here",
    "mark_entrypoint",
    "run_local",
]
