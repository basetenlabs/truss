import warnings

import pydantic

pydantic_major_version = int(pydantic.VERSION.split(".")[0])
if pydantic_major_version < 2:
    warnings.warn(
        f"Pydantic major version is less than 2 ({pydantic.VERSION}). You use Truss, "
        "but for using Truss-Chains, you must upgrade to pydantic-v2.",
        UserWarning,
    )

del pydantic
del warnings

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
