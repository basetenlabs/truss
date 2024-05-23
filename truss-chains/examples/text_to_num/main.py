# This logging is needed for debuggin class initalization.
# import logging

# log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
# date_format = "%m%d %H:%M:%S"
# logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)


import random
import string
from typing import Protocol

import pydantic
import truss_chains as chains
from sub_package import shared_chainlet
from truss import truss_config


class GenerateData(chains.ChainletBase):
    def run_remote(self, length: int) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))


MISTRAL_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_CACHE = truss_config.ModelRepo(
    repo_id=MISTRAL_HF_MODEL, allow_patterns=["*.json", "*.safetensors", ".model"]
)


class MistraLLMConfig(pydantic.BaseModel):
    hf_model_name: str


class MistralLLM(chains.ChainletBase):

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "transformers==4.38.1",
                "torch==2.0.1",
                "sentencepiece",
                "accelerate",
            ]
        ),
        compute=chains.Compute(cpu_count=2, gpu="A10G"),
        assets=chains.Assets(cached=[MISTRAL_CACHE]),
    )
    default_user_config = MistraLLMConfig(hf_model_name=MISTRAL_HF_MODEL)

    def __init__(
        self,
        context: chains.DeploymentContext[MistraLLMConfig] = chains.depends_context(),
    ) -> None:
        import torch
        import transformers

        model_name = context.user_config.hf_model_name

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

    def run_remote(self, data: str) -> str:
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
    def __init__(self, context: chains.DeploymentContext) -> None:
        ...

    def run_remote(self, data: str) -> str:
        ...


class TextToNum(chains.ChainletBase):
    def __init__(self, mistral: MistralP = chains.depends(MistralLLM)) -> None:
        self._mistral = mistral

    def run_remote(self, data: str) -> int:
        number = 0
        generated_text = self._mistral.run_remote(data)
        for char in generated_text:
            number += ord(char)

        return number


@chains.mark_entrypoint
class ExampleChain(chains.ChainletBase):
    def __init__(
        self,
        data_generator: GenerateData = chains.depends(GenerateData),
        splitter: shared_chainlet.SplitText = chains.depends(shared_chainlet.SplitText),
        text_to_num=chains.depends(TextToNum),
    ) -> None:
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num

    async def run_remote(
        self, length: int, num_partitions: int
    ) -> tuple[int, str, int]:
        data = self._data_generator.run_remote(length)
        text_parts, number = await self._data_splitter.run_remote(
            shared_chainlet.SplitTextInput(
                data=data,
                num_partitions=num_partitions,
                mode=shared_chainlet.Modes.MODE_1,
            ),
            extra_arg=123,
        )
        value = 0
        for part in text_parts.parts:
            value += self._text_to_num.run_remote(part)
        return value, data, number


if __name__ == "__main__":
    """
    Deploy remotely as:
    ```
    truss chain deploy truss-chains/examples/text_to_num/main.py ExampleChain
    ```
    """

    import asyncio

    class FakeMistralLLM:
        def run_remote(self, data: str) -> str:
            return data.upper()

    with chains.run_local():
        text_to_num_chainlet = TextToNum(mistral=FakeMistralLLM())
        wf = ExampleChain(text_to_num=text_to_num_chainlet)
        tmp = asyncio.run(wf.run_remote(length=123, num_partitions=123))
        print(tmp)
