import logging

log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
date_format = "%m%d %H:%M:%S"
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)


import random
import string
import subprocess
from typing import Protocol

import pydantic
import slay
from user_package import shared_processor

IMAGE_COMMON = slay.Image().pip_requirements_txt("common_requirements.txt")


class GenerateData(slay.ProcessorBase):

    default_config = slay.Config(image=IMAGE_COMMON)

    def run(self, length: int) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))


IMAGE_TRANSFORMERS_GPU = (
    slay.Image()
    .cuda("12.8")
    .pip_requirements_txt("common_requirements.txt")
    .pip_install("transformers")
)


class MistraLLMConfig(pydantic.BaseModel):
    hf_model_name: str


class MistralLLM(slay.ProcessorBase[MistraLLMConfig]):

    default_config = slay.Config(
        image=IMAGE_TRANSFORMERS_GPU,
        resources=slay.Resources().cpu(12).gpu("A100"),
        user_config=MistraLLMConfig(hf_model_name="EleutherAI/mistral-6.7B"),
    )
    # default_config = slay.Config(config_path="mistral_config.yaml")

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        super().__init__(context)
        # try:
        #     subprocess.check_output(["nvidia-smi"], text=True)
        # except:
        #     raise RuntimeError(
        #         f"Cannot run `{self.__class__}`, because host has no CUDA."
        #     )
        # import transformers

        # model_name = self.user_config.hf_model_name
        # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        # model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        # self._model = transformers.pipeline(
        #     "text-generation", model=model, tokenizer=tokenizer
        # )

    def run(self, data: str) -> str:
        return data.upper()
        # return self._model(data, max_length=50)


class MistralP(Protocol):
    def __init__(self, context: slay.Context) -> None:
        ...

    def run(self, data: str) -> str:
        ...


class TextToNum(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
        mistral: MistralP = slay.provide(MistralLLM),
    ) -> None:
        super().__init__(context)
        self._mistral = mistral

    def run(self, data: str) -> int:
        number = 0
        generated_text = self._mistral.run(data)
        for char in generated_text:
            number += ord(char)

        return number


class Workflow(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
        data_generator: GenerateData = slay.provide(GenerateData),
        splitter: shared_processor.SplitText = slay.provide(shared_processor.SplitText),
        text_to_num: TextToNum = slay.provide(TextToNum),
    ) -> None:
        super().__init__(context)
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run(self, length: int, num_partitions: int) -> tuple[int, str, int]:
        data = self._data_generator.run(length)
        text_parts, number = await self._data_splitter.run(data, num_partitions)
        value = 0
        for part in text_parts:
            value += self._text_to_num.run(part)
        return value, data, number


if __name__ == "__main__":
    slay.deploy_remotely([Workflow])
