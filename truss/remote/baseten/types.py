from enum import Enum

import pydantic


class ChainletData(pydantic.BaseModel):
    name: str
    oracle_version_id: str
    is_entrypoint: bool


class ModelOrigin(Enum):
    BASETEN = "BASETEN"
    CHAINS = "CHAINS"
