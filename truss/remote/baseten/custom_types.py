import pathlib
import sys
from enum import Enum
from typing import Optional

import pydantic

import truss


class DeployedChainlet(pydantic.BaseModel):
    name: str
    is_entrypoint: bool
    is_draft: bool
    status: str
    logs_url: str
    oracle_name: str


class ChainletArtifact(pydantic.BaseModel):
    truss_dir: pathlib.Path
    display_name: str
    name: str


class ModelOrigin(Enum):
    BASETEN = "BASETEN"
    CHAINS = "CHAINS"


class OracleData(pydantic.BaseModel):
    class Config:
        protected_namespaces = ()

    model_name: str
    s3_key: str
    encoded_config_str: str
    semver_bump: Optional[str] = "MINOR"
    version_name: Optional[str] = None


# This corresponds to `ChainletInputAtomicGraphene` in the backend.
class ChainletDataAtomic(pydantic.BaseModel):
    name: str
    oracle: OracleData


class TrussUserEnv(pydantic.BaseModel):
    truss_client_version: str
    python_version: str
    pydantic_version: str
    mypy_version: Optional[str]

    @classmethod
    def collect(cls):
        py_version = sys.version_info
        try:
            import mypy.version

            mypy_version = mypy.version.__version__
        except ImportError:
            mypy_version = None

        return cls(
            truss_client_version=truss.version(),
            python_version=f"{py_version.major}.{py_version.minor}.{py_version.micro}",
            pydantic_version=pydantic.version.version_short(),
            mypy_version=mypy_version,
        )
