import asyncio
from typing import Any, Dict, List


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        pass

    async def predict(self, model_input: Any) -> Dict[str, List]:
        await asyncio.sleep(1)
        n = 10
        i = 0
        try:
            for i in range(n):
                await asyncio.sleep(1.0)
                print(i)
                yield str(i)
        finally:
            if i < n:
                print(f"Generation stopped in iteration {i}")
