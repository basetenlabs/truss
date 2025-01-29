from typing import Dict


class Model:
    def __init__(self):
        pass

    def load(self):
        self._predict_count = 0
        self._completions_count = 0
        self._chat_completions_count = 0

    async def predict(self, inputs: Dict) -> int:
        self._predict_count += inputs["increment"]
        return self._predict_count

    async def completions(self, inputs: Dict) -> int:
        self._completions_count += inputs["increment"]
        return self._completions_count

    async def chat_completions(self, inputs: Dict) -> int:
        self._chat_completions_count += inputs["increment"]
        return self._chat_completions_count
