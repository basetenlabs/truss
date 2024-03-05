import slay
from slay import stub
from user_dependencies import IMAGE_COMMON, Parameters, WorkflowResult

from . import dependencies


class Workflow(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        data_generator = stub.stub_factory(dependencies.GenerateData, context)
        splitter = stub.stub_factory(dependencies.SplitText, context)
        text_to_num = stub.stub_factory(dependencies.TextToNum, context)
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
