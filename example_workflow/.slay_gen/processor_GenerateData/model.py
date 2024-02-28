import logging
import random
import string

import pydantic
import slay
from user_package import shared_processor


class Parameters(pydantic.BaseModel):
    length: int = 100
    num_partitions: int = 4
    num_replications: int = 3


class GenerateData(slay.BaseProcessor):
    default_config = slay.Config(name="MyDataGenerator")

    def gen_data(self, params: Parameters) -> str:
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=params.length)
        )


class TextReplicator(slay.BaseProcessor):
    def replicate(self, data: str, params: Parameters) -> str:
        return str(data * params.num_replications)


class TextToNum(slay.BaseProcessor):
    _replicator: TextReplicator

    def __init__(
        self,
        config: slay.Config,
        replicator: TextReplicator = slay.provide(TextReplicator),
    ) -> None:
        super().__init__(config)
        self._replicator = replicator

    def to_num(self, data: str, params: Parameters) -> int:
        number = 0
        replicator_result = self._replicator.replicate(data, params)
        for char in replicator_result:
            number += ord(char)
        return number


class Workflow(slay.BaseProcessor):
    def __init__(
        self,
        config: slay.Config = slay.provide_config(),
        data_generator: GenerateData = slay.provide(GenerateData),
        data_splitter: shared_processor.SplitData = slay.provide(
            shared_processor.SplitData
        ),
        text_to_num: TextToNum = slay.provide(TextToNum),
    ) -> None:
        super().__init__(config)
        self._data_generator = data_generator
        self._data_splitter = data_splitter
        self._text_to_num = text_to_num

    def run(self, params: Parameters) -> int:
        data = self._data_generator.gen_data(params)
        text_parts = self._data_splitter.split(data, params)
        value = 0
        for part in text_parts:
            value += self._text_to_num.to_num(part, params)
        return value
