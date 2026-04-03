from typing import Any, Dict


class Model:
    def __init__(self, **kwargs) -> None:
        pass

    def load(self):
        pass

    def predict(self, model_input: Any) -> Dict[str, bool]:
        return {"success": True}
