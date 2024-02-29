from slay import framework
from typing import Protocol
import pydantic

# Copy I/O defs, e.g. Parameters
class Parameters(pydantic.BaseModel):
    length: int = 100
    num_partitions: int = 4

# Generate a Protocol to replace fixed processor types with.
class GenerateDataP(Protocol):

    def gen_data(self, params: Parameters) -> str:
        ...

# Generate the actual stub.
# Question: should url be hardcoded or passed via argument?
class GenerateData:

    def __init__(self, url: str, api_key: str) -> None:
        self._baseten_stub = framework.BasetenStub(url, api_key)

    def gen_data(self, params: Parameters) -> str:
        return self._baseten_stub.predict_sync(params.model_dump_json())

    
