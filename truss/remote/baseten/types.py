import pydantic


class ChainletData(pydantic.BaseModel):
    name: str
    oracle_version_id: str
    is_entrypoint: bool
