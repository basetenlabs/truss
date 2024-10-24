from typing import Protocol

from truss.base import truss_config

import truss_chains as chains

MISTRAL_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


class MistralLLM(chains.ChainletBase):
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
        compute=chains.Compute(cpu_count=2, gpu="A10G"),
        # Cache the model weights in the image and make the huggingface
        # access token secret available to the model.
        assets=chains.Assets(
            cached=[
                truss_config.ModelRepo(
                    repo_id=MISTRAL_HF_MODEL,
                    allow_patterns=["*.json", "*.safetensors", ".model"],
                )
            ],
            secret_keys=["hf_access_token"],
        ),
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

    async def run_remote(self, data: str) -> str:
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
    async def run_remote(self, data: str) -> str: ...


@chains.mark_entrypoint
class PoemGenerator(chains.ChainletBase):
    def __init__(self, mistral_llm: MistralP = chains.depends(MistralLLM)) -> None:
        self._mistral_llm = mistral_llm

    async def run_remote(self, words: list[str]) -> list[str]:
        tasks = []
        for word in words:
            prompt = f"Generate a poem about: {word}"
            tasks.append(asyncio.ensure_future(self._mistral_llm.run_remote(prompt)))
        return list(await asyncio.gather(*tasks))


if __name__ == "__main__":
    import asyncio

    class FakeMistralLLM:
        async def run_remote(self, data: str) -> str:
            return data.upper()

    with chains.run_local():
        poem_generator = PoemGenerator(mistral_llm=FakeMistralLLM())
        poems = asyncio.get_event_loop().run_until_complete(
            poem_generator.run_remote(words=["apple", "banana"])
        )
        print(poems)
