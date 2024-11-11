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


class ChainletData(pydantic.BaseModel):
    name: str
    oracle_version_id: str
    is_entrypoint: bool


class ChainletArtifact(pydantic.BaseModel):
    truss_dir: pathlib.Path
    is_entrypoint: bool
    display_name: str
    name: str


class ModelOrigin(Enum):
    BASETEN = "BASETEN"
    CHAINS = "CHAINS"


class OracleData(pydantic.BaseModel):
    model_name: str
    s3_key: str
    encoded_config_str: str
    semver_bump: Optional[str] = "MINOR"
    is_trusted: bool
    version_name: Optional[str] = None


class ChainletDataAtomic(pydantic.BaseModel):
    name: str
    is_entrypoint: bool
    oracle: OracleData
