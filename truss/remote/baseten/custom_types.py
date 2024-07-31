from enum import Enum

import pydantic


class DeployedChainlet(pydantic.BaseModel):
    name: str
    is_entrypoint: bool
    status: str
    logs_url: str


class ChainletData(pydantic.BaseModel):
    name: str
    oracle_version_id: str
    is_entrypoint: bool


class ModelOrigin(Enum):
    BASETEN = "BASETEN"
    CHAINS = "CHAINS"
