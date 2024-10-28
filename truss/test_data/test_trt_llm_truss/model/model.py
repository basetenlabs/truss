from typing import Any

import fastapi


class Model:
    def __init__(self, trt_llm, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._engine = trt_llm["engine"]

    async def predict(self, model_input: Any, request: fastapi.Request) -> Any:
        return await self._engine.predict(model_input, request)
