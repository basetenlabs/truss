from typing import Dict


class Model:
    def __init__(self, **kwargs):
        pass

    def chat_completions(self, input: Dict) -> str:
        return "chat_completions"

    def completions(self, input: Dict) -> str:
        return "completions"

    def predict(self, input: Dict) -> str:
        return "predict"
