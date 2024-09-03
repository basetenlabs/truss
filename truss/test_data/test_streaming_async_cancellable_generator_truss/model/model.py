import asyncio
from typing import Any, Awaitable, Callable, Dict, List


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        pass

    async def predict(
        self, model_input: Any, is_cancelled_fn: Callable[[], Awaitable[bool]]
    ) -> Dict[str, List]:
        # Invoke model on model_input and calculate predictions here.
        await asyncio.sleep(1)
        if await is_cancelled_fn():
            print("Cancelled (before gen).")
            return

        for i in range(5):
            await asyncio.sleep(1.0)
            print(i)
            yield str(i)
            if await is_cancelled_fn():
                print("Cancelled (during gen).")
                return
