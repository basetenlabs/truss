r"""
A workflow combines multiple remote operations to produce a final result.

The simplest case is a plain chain:

Preprocssing (CPU) --> ML model (GPU) --> Postprocessing (CPU)

But any DAG-like structure can be realized:

    --> B --->
  /            \
A               E --> G
  \            /       \
    --> C --> D         --> I
              \           /
                F -----> H

Remote operations are conceptually the same as truss models hosted on Baseten.
"""

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

    def __init__(
        self,
        config: slay.Config = slay.provide_config(),
    ) -> None:
        super().__init__(config)
        try:
            subprocess.check_output(["nvidia-smi"], text=True)
        except:
            raise RuntimeError(
                f"Cannot run `{self.__class__}`, because host has no CUDA."
            )
        import transformers

        model_name = config.user_config.hf_model_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self._model = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )

    def llm_gen(self, data: str) -> str:
        return self._model(data, max_length=50)


class MistralP(Protocol):
    def __init__(self, config: slay.Config) -> None:
        ...

    def llm_gen(self, data: str) -> str:
        ...


class TextToNum(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        config: slay.Config = slay.provide_config(),
        mistral: MistralP = slay.provide(MistralLLM),
    ) -> None:
        super().__init__(config)
        self._mistral = mistral

    def to_num(self, data: str, params: Parameters) -> int:
        number = 0
        generated_text = self._mistral.llm_gen(data)
        for char in generated_text:
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
    log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
    date_format = "%m%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)

    # Local test or dev execution - context manager makes sure local processors
    # are instantiated and injected.
    # with slay.run_local():
    #     wf = Workflow()
    #     params = Parameters()
    #     result = wf.run(params=params)
    #     print(result)

    class FakeMistralLLM(slay.ProcessorBase):
        def llm_gen(self, data: str) -> str:
            return data.upper()

    with slay.run_local():
        text_to_num = TextToNum(mistral=FakeMistralLLM())
        wf = Workflow(text_to_num=text_to_num)
        params = Parameters()
        result = wf.run(params=params)
        print(result)

    # Gives a `UsageError`, because not in `run_local` context.
    try:
        wf = Workflow()
    except slay.UsageError as e:
        print(e)

    # A "marker" to designate which processors should be deployed as public remote
    # service points. Depenedency processors will also be deployed, but only as
    # "internal" services, not as a "public" sevice endpoint.
    slay.deploy_remotely([Workflow])
