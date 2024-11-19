import pathlib
from enum import Enum
from typing import Optional

import pydantic


class DeployedChainlet(pydantic.BaseModel):
    name: str
    is_entrypoint: bool
    is_draft: bool
    status: str
    logs_url: str
    oracle_predict_url: str
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
    is_trusted: bool
    version_name: Optional[str] = None


# This corresponds to `ChainletInputAtomicGraphene` in the backend.
class ChainletDataAtomic(pydantic.BaseModel):
    name: str
    oracle: OracleData
