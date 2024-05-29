# This logging is needed for debuggin class initalization.
# import logging

# log_format = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s"
# date_format = "%m%d %H:%M:%S"
# logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)


import random
import string
from typing import Protocol

import truss_chains as chains
from sub_package import shared_chainlet
from truss import truss_config


class GenerateData(chains.ChainletBase):
    def run_remote(self, length: int) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))


MISTRAL_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# This configures to cache model weights from the hunggingface repo
# in the docker image that is used for deploying the Chainlet.
MISTRAL_CACHE = truss_config.ModelRepo(
    repo_id=MISTRAL_HF_MODEL, allow_patterns=["*.json", "*.safetensors", ".model"]
)


class MistralLLM(chains.ChainletBase):
    # The RemoteConfig object defines the resources required for this chainlet.
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            # The mistral model needs some extra python packages.
            pip_requirements=[
                "transformers==4.38.1",
                "torch==2.0.1",
                "sentencepiece",
                "accelerate",
            ]
        ),
        # The mistral model needs a GPU and more CPUs.
        compute=chains.Compute(cpu_count=2, gpu="A10G"),
        # Cache the model weights in the image and make the huggingface
        # access token secret available to the model.
        assets=chains.Assets(cached=[MISTRAL_CACHE], secret_keys=["hf_access_token"]),
    )

    def __init__(
        self,
        # Adding the `context` to the init arguments, allows us to access the
        # huggingface token.
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        # Note the imports of the *specific* python requirements are pushed down to
        # here. This code will only be executed on the remotely deployed chainlet,
        # not in the local environment, so we don't need to install these packages
        # in the local dev environment.
        import torch
        import transformers

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            MISTRAL_HF_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=context.secrets["hf_access_token"],
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            MISTRAL_HF_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=context.secrets["hf_access_token"],
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
    def run_remote(self, data: str) -> str:
        ...


@chains.mark_entrypoint
class PoemGenerator(chains.ChainletBase):
    def __init__(self, mistral_llm: MistralP = chains.depends(MistralLLM)) -> None:
        self._mistral_llm = mistral_llm

    def run_remote(self, words: list[str]) -> list[str]:
        results = []
        for word in words:
            poem = self._mistral_llm.run_remote(f"Generate a poem about: {word}")
            results.append(poem)
        return results


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
