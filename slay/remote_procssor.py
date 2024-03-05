import pathlib

from slay import definitions
from truss.server.shared import secrets_resolver


class BasetenModel:
    _config: dict
    _data_dir: pathlib.Path
    _secrets: secrets_resolver.Secrets

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        self._config = config
        self._data_dir = data_dir
        self._secrets = secrets

    @property
    def _baseten_api_key(self) -> str:
        return self._secrets.get("baseten_api_key")

    def _converted_config(self):
        # TODO
        return definitions.Config()

    # Generate something like this (sync|async variants):
    """
    def load(self) -> None:
        super().___init__(
            config=self._converted_config(),
            data_generator=generated_stubs_sync.GenerateData(self._baseten_api_key),
            data_splitter=generated_stubs_sync.SplitData(self._baseten_api_key),
            text_to_num=generated_stubs_sync.TextToNum(self._baseten_api_key),
        )
    """
    # For predict sync|async variants.
    # If endpoint as pydantic types, use them in siganture, otherwise use `Any`
    # and expect that it's either a trivial value or a tuple of that.
    """
    async def predict(self, payload: Any) -> Any: ...
        return await super().run(payload)

    def predict(self, payload: Any) -> Any: ...
        return super().run(payload)

    async def predict(self, payload: PydanticParams) -> PydanticOutput: ...
        return await super().run(payload)
    """
