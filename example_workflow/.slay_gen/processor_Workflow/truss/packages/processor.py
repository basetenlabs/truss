import logging
import pathlib

from slay import definitions, stub
from truss.templates.shared import secrets_resolver

from . import user_stubs

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


class Parameters(pydantic.BaseModel):
    length: int = 100
    num_partitions: int = 4


class WorkflowResult(pydantic.BaseModel):
    number: int
    params: Parameters


class GenerateData(slay.ProcessorBase):

    default_config = slay.Config(image=IMAGE_COMMON)

    def run(self, params: Parameters) -> str:
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
    # default_config = slay.Config(config_path="mistral_config.yaml")

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        super().__init__(context)
        try:
            subprocess.check_output(["nvidia-smi"], text=True)
        except:
            raise RuntimeError(
                f"Cannot run `{self.__class__}`, because host has no CUDA."
            )
        import transformers

        model_name = self.user_config.hf_model_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self._model = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )

    def run(self, data: str) -> str:
        return self._model(data, max_length=50)


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

    def run(self, data: str, params: Parameters) -> int:
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
    ) -> None:
        data_generator = stub.stub_factory(user_stubs.GenerateData, context)
        splitter = stub.stub_factory(user_stubs.SplitText, context)
        text_to_num = stub.stub_factory(user_stubs.TextToNum, context)
        super().__init__(context)
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run(self, params: Parameters, num: int) -> tuple[WorkflowResult, int]:
        data = self._data_generator.run(params)
        text_parts = await self._data_splitter.run(data, params.num_partitions)
        value = 0
        for part in text_parts:
            value += self._text_to_num.run(part, params)
        return WorkflowResult(number=value, params=params), value


if __name__ == "__main__":
    import asyncio

    # Local test or dev execution - context manager makes sure local processors
    # are instantiated and injected.
    # with slay.run_local():
    #     wf = Workflow()
    #     params = Parameters()
    #     result = wf.run(params=params)
    #     print(result)

    class FakeMistralLLM(slay.ProcessorBase):
        def run(self, data: str) -> str:
            return data.upper()

    with slay.run_local():
        text_to_num = TextToNum(mistral=FakeMistralLLM())
        wf = Workflow(text_to_num=text_to_num)
        params = Parameters()
        result = asyncio.run(wf.run(params=params, num=123))
        print(result)

    # # Gives a `UsageError`, because not in `run_local` context.
    # try:
    #     wf = Workflow()
    # except slay.UsageError as e:
    #     print(e)

    # A "marker" to designate which processors should be deployed as public remote
    # service points. Depenedency processors will also be deployed, but only as
    # "internal" services, not as a "public" sevice endpoint.
    slay.deploy_remotely([Workflow])


class Model:
    _context: definitions.Context
    _processor: Workflow

    def __init__(
        self, config: dict, data_dir: pathlib.Path, secrets: secrets_resolver.Secrets
    ) -> None:
        truss_metadata = definitions.TrussMetadata.parse_obj(
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
            params=Parameters.parse_obj(payload["params"]), num=payload["num"]
        )
        return (result[0].dict(), result[1])
