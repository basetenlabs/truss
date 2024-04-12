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

IMAGE_COMMON = slay.DockerImage().pip_requirements_file(
    slay.make_abs_path_here("requirements.txt")
)


class GenerateData(slay.ProcessorBase):

    remote_config = slay.RemoteConfig(docker_image=IMAGE_COMMON)

    def run(self, length: int) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))


IMAGE_TRANSFORMERS_GPU = (
    slay.DockerImage()
    .pip_requirements_file(slay.make_abs_path_here("requirements.txt"))
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

    remote_config = slay.RemoteConfig(
        docker_image=IMAGE_TRANSFORMERS_GPU,
        compute=slay.Compute().cpu(2).gpu("A10G"),
        assets=slay.Assets().cached([MISTRAL_CACHE]),
    )
    default_user_config = MistraLLMConfig(hf_model_name=MISTRAL_HF_MODEL)

    def __init__(
        self,
        context: slay.DeploymentContext[MistraLLMConfig] = slay.provide_context(),
    ) -> None:
        super().__init__(context)
        import torch
        import transformers

        model_name = self.user_config.hf_model_name

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self._generate_args = {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

    def run(self, data: str) -> str:
        import torch

        formatted_prompt = f"[INST] {data} [/INST]"
        input_ids = self._tokenizer(
            formatted_prompt, return_tensors="pt"
        ).input_ids.cuda()
        with torch.no_grad():
            output = self._model.generate(inputs=input_ids, **self._generate_args)
            result = self._tokenizer.decode(output[0])
        return result


class MistralP(Protocol):
    def __init__(self, context: slay.DeploymentContext) -> None:
        ...

    def run(self, data: str) -> str:
        ...


class TextToNum(slay.ProcessorBase):
    remote_config = slay.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.DeploymentContext = slay.provide_context(),
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
    remote_config = slay.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        context: slay.DeploymentContext = slay.provide_context(),
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
        text_parts, number = await self._data_splitter.run(
            shared_processor.SplitTextInput(
                data=data,
                num_partitions=num_partitions,
                mode=shared_processor.Modes.MODE_1,
            ),
            extra_arg=123,
        )
        value = 0
        for part in text_parts.parts:
            value += self._text_to_num.run(part)
        return value, data, number


if __name__ == "__main__":
    import logging

    from slay import utils

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
    date_format = "%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # class FakeMistralLLM(slay.ProcessorBase):
    #     def run(self, data: str) -> str:
    #         return data.upper()
    #
    # import asyncio
    #
    # with slay.run_local():
    #     text_to_num = TextToNum(mistral=FakeMistralLLM())
    #     wf = Workflow(text_to_num=text_to_num)
    #     tmp = asyncio.run(wf.run(length=123, num_partitions=123))
    #     print(tmp)

    with utils.log_level(logging.DEBUG):
        remote = slay.deploy_remotely(
            Workflow, workflow_name="Test", only_generate_trusses=True, publish=False
        )

    # response = utils.call_workflow_dbg(remote, {"length": 1000, "num_partitions": 100})
    # print(response)
    # print(response.json())
