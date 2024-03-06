import pathlib

import slay
from slay import definitions
from truss.server.shared import secrets_resolver
from user_dependencies import IMAGE_NUMPY


class SplitText(slay.ProcessorBase):

    default_config = slay.Config(image=IMAGE_NUMPY)

    async def split(self, data: str, num_partitions: int) -> list[str]:
        import numpy as np

        parts = np.array_split(np.array(list(data)), 3)
        return ["".join(part) for part in parts]


class Model:
    _context: definitions.Context
    _processor: SplitText

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        truss_metadata = definitions.TrussMetadata.model_validate(
            config["model_metadata"]["slay_metadata"]
        )
        self._context = definitions.Context(
            user_config=truss_metadata.user_config,
            stub_cls_to_url=truss_metadata.stub_cls_to_url,
            secrets=secrets,
        )

    def load(self) -> None:
        self._processor = SplitText(context=self._context)

    async def predict(self, payload):
        result = await self._processor.split(
            data=payload["data"], num_partitions=payload["num_partitions"]
        )
        return result
