from typing import Protocol

import pydantic
from slay import framework


# Copy I/O defs, e.g. Parameters
class Parameters(pydantic.BaseModel):
    length: int = 100
    num_partitions: int = 4


class WorkflowResult(pydantic.BaseModel):
    number: int
    params: Parameters


# Generate a Protocol to replace fixed processor types with.
class WorkflowP(Protocol):

    def run(self, params: Parameters) -> WorkflowResult: ...


# Generate the actual stub.
# Question: should url be hardcoded or passed via argument?
class GenerateData:

    def __init__(self, url: str, api_key: str) -> None:
        self._remote = framework.BasetenStub(url, api_key)

    def gen_data(self, params: Parameters) -> WorkflowResult:
        json_args = params.model_dump_json()
        json_result = self._remote.predict_sync(json_args)
        result = WorkflowResult.model_validate(json_result)
        return result