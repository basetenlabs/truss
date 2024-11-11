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


class ModelOrigin(Enum):
    BASETEN = "BASETEN"
    CHAINS = "CHAINS"


class OracleData(pydantic.BaseModel):
    name: str
    s3_key: str
    config: str
    semver_bump: Optional[str] = "MINOR"
    client_version: Optional[str]
    is_trusted: bool
    deployment_name: Optional[str] = None
    origin: Optional[ModelOrigin] = None
    environment: Optional[str] = None


class ChainletDataAtomic(pydantic.BaseModel):
    name: str
    is_entrypoint: bool
    oracle: OracleData
