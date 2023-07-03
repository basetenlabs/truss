from typing import Any

from vllm import LLM


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.llm = None

    def load(self):
        self.llm = LLM(model="lmsys/vicuna-7b-v1.3")

    def predict(self, model_input: Any) -> Any:
        return self.llm.generate(**model_input)
