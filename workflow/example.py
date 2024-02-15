import random
import string

import pydantic
from workflow import slay

########################################################################################


class Parameters(pydantic.BaseModel):
    length: int = 100
    num_partitions: int = 4
    num_replications: int = 3


class WithParamsBase(pydantic.BaseModel):
    params: Parameters


class GenerateDataInput(WithParamsBase):
    ...


class GenerateDataOutput(WithParamsBase):
    text: str


class GenerateData(slay.BaseProcessor):

    default_config = slay.Config(name="MyDataGenerator")

    def gen_data(self, data: GenerateDataInput) -> GenerateDataOutput:
        return GenerateDataOutput(
            params=data.params,
            text="".join(
                random.choices(
                    string.ascii_letters + string.digits, k=data.params.length
                )
            ),
        )


########################################################################################


class SplitDataInput(WithParamsBase):
    text: str


class SplitDataOutput(WithParamsBase):
    text_parts: list[str]


class SplitData(slay.BaseProcessor):
    def split(self, data: SplitDataInput) -> SplitDataOutput:
        num_partitions = data.params.num_partitions
        part_length = len(data.text) // num_partitions
        text_parts = [
            data.text[i * part_length : (i + 1) * part_length]
            + (
                data.text[num_partitions * part_length :]
                if i == num_partitions - 1
                else ""
            )
            for i in range(num_partitions)
        ]
        return SplitDataOutput(params=data.params, text_parts=text_parts)


########################################################################################


class TextReplicatorInput(WithParamsBase):
    text: str


class TextReplicatorOutput(WithParamsBase):
    text: str


class TextReplicator(slay.BaseProcessor):
    def replicate(self, data: TextReplicatorInput) -> TextReplicatorOutput:
        return TextReplicatorOutput(
            params=data.params, text=str(data.text * data.params.num_replications)
        )


########################################################################################


class TextToNumInput(WithParamsBase):
    text: str


class TextToNumOutput(WithParamsBase):
    number: int


class TextToNum(slay.BaseProcessor):

    _replicator: TextReplicator

    def __init__(
        self,
        config: slay.Config,
        replicator: TextReplicator = slay.provide(TextReplicator),
    ) -> None:
        super().__init__(config)
        self._replicator = replicator

    def to_num(self, data: TextToNumInput) -> TextToNumOutput:
        number = 0
        replicator_result = self._replicator.replicate(
            TextReplicatorInput(params=data.params, text=data.text)
        )
        for char in replicator_result.text:
            number += ord(char)

        return TextToNumOutput(params=data.params, number=number)


########################################################################################


class WorkflowInput(WithParamsBase):
    ...


class WorkflowOutput(pydantic.BaseModel):
    number: int


class Workflow(slay.BaseProcessor):

    # _data_generator: GenerateData
    # _splitter:

    def __init__(
        self,
        config: slay.Config = slay.provide_config(),
        data_generator: GenerateData = slay.provide(GenerateData),
        data_splitter: SplitData = slay.provide(SplitData),
        text_to_num: TextToNum = slay.provide(TextToNum),
    ) -> None:
        super().__init__(config)
        self._data_generator = data_generator
        self._data_splitter = data_splitter
        self._text_to_num = text_to_num

    def run(self, wf_input: WorkflowInput) -> WorkflowOutput:
        gen_result = self._data_generator.gen_data(
            GenerateDataInput(params=wf_input.params)
        )
        parts_result = self._data_splitter.split(
            SplitDataInput(params=wf_input.params, text=gen_result.text)
        )
        value = 0
        for part in parts_result.text_parts:
            num_result = self._text_to_num.to_num(
                TextToNumInput(params=wf_input.params, text=part)
            )
            value += num_result.number

        return WorkflowOutput(number=value)


if __name__ == "__main__":

    # Local test or dev execution - context manager makes sure local processors
    # are instantiated and injected.
    with slay.run_local():
        wf = Workflow()
        params = Parameters()
        result = wf.run(WorkflowInput(params=params))
        print(result)

    # A "marker" to designate which processors should be deployed as public remote
    # service points. Depenedency processors will also be deployed, but only as
    # "internal" services, not as a "public" sevice endpoint.
    slay.deploy_remotely([Workflow])
