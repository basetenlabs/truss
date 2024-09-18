from typing import Any, Dict, List


class Model:
    async def predict(self, model_input: Any) -> Dict[str, List]:
        for i in range(100):
            yield str(i)
