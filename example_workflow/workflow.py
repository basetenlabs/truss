import logging
import random
import string
import subprocess
from typing import Protocol

import pydantic
import slay
from user_package import shared_processor

IMAGE_COMMON = slay.Image().pip_requirements_txt("common_requirements.txt")


class Parameters(pydantic.BaseModel):
    length: int = 100
    num_partitions: int = 4


class WorkflowResult(pydantic.BaseModel):
    number: int
    params: Parameters


class GenerateData(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def gen_data(self, params: Parameters) -> str:
        return "".join(
            random.choices(string.ascii_letters + string.digits, k=params.length)
        )


class TextToNum(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def to_num(self, data: str, params: Parameters) -> int:
        number = 0
        for char in data:
            number += ord(char)

        return number


class Workflow(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        config: slay.Config = slay.provide_config(),
        data_generator: GenerateData = slay.provide(GenerateData),
        splitter: shared_processor.SplitText = slay.provide(shared_processor.SplitText),
        text_to_num: TextToNum = slay.provide(TextToNum),
    ) -> None:
        super().__init__(config)
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    def run(self, params: Parameters) -> WorkflowResult:
        data = self._data_generator.gen_data(params)
        text_parts = self._data_splitter.split(data, params.num_partitions)
        value = 0
        for part in text_parts:
            value += self._text_to_num.to_num(part, params)
        return WorkflowResult(number=value, params=params)


if __name__ == "__main__":
    params = Parameters(length=100, num_partitions=5)

    wf = Workflow()
    print(wf.run(params))
