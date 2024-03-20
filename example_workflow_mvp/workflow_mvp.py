# This logging is needed for debuggin class initalization.
# import logging

# log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
# date_format = "%m%d %H:%M:%S"
# logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)


import random
import string
from typing import Protocol

import pydantic
import slay
from truss import truss_config
from user_package import shared_processor

IMAGE_COMMON = slay.Image().pip_requirements_file(
    "/home/marius-baseten/workbench/truss/example_workflow_mvp/requirements.txt"
)


class GenerateData(slay.ProcessorBase):

    default_config = slay.Config(image=IMAGE_COMMON)

    def run(self, length: int) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))


IMAGE_TRANSFORMERS_GPU = (
    slay.Image()
    .pip_requirements_file(
        "/home/marius-baseten/workbench/truss/example_workflow_mvp/requirements.txt"
    )
    .pip_requirements(
        ["transformers==4.38.1", "torch==2.0.1", "sentencepiece", "accelerate"]
    )
)


MISTRAL_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_CACHE = truss_config.ModelRepo(
    repo_id=MISTRAL_HF_MODEL, allow_patterns=["*.json", "*.safetensors", ".model"]
)


class MistraLLMConfig(pydantic.BaseModel):
    hf_model_name: str


class MistralLLM(slay.ProcessorBase[MistraLLMConfig]):

    default_config = slay.Config(
        image=IMAGE_TRANSFORMERS_GPU,
        compute=slay.Compute().cpu(2).gpu("A10G"),
        assets=slay.Assets().cached([MISTRAL_CACHE]),
        user_config=MistraLLMConfig(hf_model_name=MISTRAL_HF_MODEL),
    )
    # default_config = slay.Config(config_path="mistral_config.yaml")

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        super().__init__(context)
        # import subprocess
        # try:
        #     subprocess.check_output(["nvidia-smi"], text=True)
        # except:
        #     raise RuntimeError(
        #         f"Cannot run `{self.__class__}`, because host has no CUDA."
        #     )
        import transformers

        model_name = self.user_config.hf_model_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self._model = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )

    def run(self, data: str) -> str:
        # return data.upper()
        result = self._model(data, max_length=50)
        print(result)
        return result


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
    import logging

    # from slay import utils
    # from slay.truss_compat import deploy

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
    date_format = "%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # class FakeMistralLLM(slay.ProcessorBase):
    #     def run(self, data: str) -> str:
    #         return data.upper()

    # import asyncio
    # with slay.run_local():
    #     text_to_num = TextToNum(mistral=FakeMistralLLM())
    #     wf = Workflow(text_to_num=text_to_num)
    #     result = asyncio.run(wf.run(length=123, num_partitions=123))
    #     print(result)

    remote = slay.deploy_remotely(Workflow, generate_only=False)

    # remote = slay.definitions.BasetenRemoteDescriptor(
    #     b10_model_id="7qk59gdq",
    #     b10_model_version_id="woz52g3",
    #     b10_model_name="Workflow",
    #     b10_model_url="https://model-7qk59gdq.api.baseten.co/production",
    # )
    # with utils.log_level(logging.INFO):
    #     response = deploy.call_workflow_dbg(
    #         remote, {"length": 1000, "num_partitions": 100}
    #     )
    # print(response)
    # print(response.json())
