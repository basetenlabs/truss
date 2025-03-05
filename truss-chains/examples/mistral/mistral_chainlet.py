import asyncio
from typing import Protocol

import truss_chains as chains
from truss.base import truss_config

MISTRAL_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


class MistralLLM(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "transformers",
                "torch==2.0.1",
                "sentencepiece",
                "accelerate",
                "tokenizers",
            ]
        ),
        compute=chains.Compute(cpu_count=2, gpu="A10G"),
        assets=chains.Assets(
            # Cache the model weights in the image and make the
            cached=[
                truss_config.ModelRepo(
                    repo_id=MISTRAL_HF_MODEL,
                    allow_patterns=["*.json", "*.safetensors", ".model"],
                )
            ],
            # Make huggingface access token secret available to the model.
            secret_keys=["hf_access_token"],
        ),
    )

    def __init__(
        self,
        # Using the optional `context` init-argument, allows to access secrets
        #  such as the huggingface token.
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        # Note: the imports of the *Chainlet-specific* python requirements are pushed
        # down to here (so you don't need them locally or in the other Chainlet).
        # The code here is only executed on the remotely deployed chainlet, where
        # these dependencies are included in the docker image.
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

    async def run_remote(self, data: str) -> str:
        import torch

        prompt = f"[INST] {data} [/INST]"
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            output = self._model.generate(inputs=input_ids, **self._generate_args)
            result = self._tokenizer.decode(output[0])
        return (
            result.replace(prompt, "")
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("\n", " ")
            .strip()
        )


class MistralP(Protocol):
    async def run_remote(self, data: str) -> str: ...


@chains.mark_entrypoint
class PoemGenerator(chains.ChainletBase):
    def __init__(self, mistral_llm: MistralP = chains.depends(MistralLLM)) -> None:
        self._mistral_llm = mistral_llm

    async def run_remote(self, words: list[str]) -> dict[str, str]:
        tasks = []
        for word in words:
            prompt = f"Write a really short poem about: {word}"
            tasks.append(asyncio.ensure_future(self._mistral_llm.run_remote(prompt)))

        poems = list(await asyncio.gather(*tasks))
        return {word: poem for word, poem in zip(words, poems)}


if __name__ == "__main__":
    import asyncio

    class FakeMistralLLM:
        async def run_remote(self, data: str) -> str:
            return data.upper()

    with chains.run_local():
        poem_generator = PoemGenerator(mistral_llm=FakeMistralLLM())
        results = asyncio.get_event_loop().run_until_complete(
            poem_generator.run_remote(words=["apple", "banana"])
        )
        print(results)
