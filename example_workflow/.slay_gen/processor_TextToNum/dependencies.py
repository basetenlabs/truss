from typing import Protocol

from slay import stub


class MistralLLMP(Protocol):
    def llm_gen(self, data: str) -> str:
        ...


class MistralLLM(stub.StubBase):
    def __init__(self, url: str, api_key: str) -> None:
        self._remote = stub.BasetenSession(url, api_key)

    def llm_gen(self, data: str) -> str:
        json_args = [data]
        json_result = self._remote.predict_sync(json_args)
        return json_result
