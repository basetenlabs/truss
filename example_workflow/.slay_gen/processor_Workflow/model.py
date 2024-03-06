import pathlib

import slay
from slay import definitions, stub
from truss.server.shared import secrets_resolver
from user_dependencies import IMAGE_COMMON, Parameters, WorkflowResult, params

from . import user_stubs


class Workflow(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        data_generator = stub.stub_factory(user_stubs.GenerateData, context)
        splitter = stub.stub_factory(user_stubs.SplitText, context)
        text_to_num = stub.stub_factory(user_stubs.TextToNum, context)
        super().__init__(context)
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run(self, params: Parameters) -> tuple[WorkflowResult, int]:
        data = self._data_generator.gen_data(params)
        text_parts = await self._data_splitter.split(data, params.num_partitions)
        value = 0
        for part in text_parts:
            value += self._text_to_num.to_num(part, params)
        return WorkflowResult(number=value, params=params), value


class Model:
    _context: definitions.Context
    _processor: Workflow

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
        self._processor = Workflow(context=self._context)

    async def predict(self, payload):
        result = await self._processor.run(
            params=Parameters.model_validate(payload["params"])
        )
        return (result[0].model_dump(), result[1])
