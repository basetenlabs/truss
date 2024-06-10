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
        f"Pydantic v2 is not supported for `Truss-Chains`. Please upgrade to v2. "
        "You can still use other Truss functionality."
    )

del pydantic


# flake8: noqa F401
from truss_chains.definitions import (
    Assets,
    ChainsRuntimeError,
    Compute,
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
    deploy_remotely,
    mark_entrypoint,
    run_local,
)
from truss_chains.stub import StubBase
from truss_chains.utils import make_abs_path_here
