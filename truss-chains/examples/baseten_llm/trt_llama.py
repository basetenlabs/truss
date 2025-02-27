from typing import AsyncIterator

from truss.base import trt_llm_config, truss_config

import truss_chains as chains
from truss_chains import baseten_llm


class Llama7BChainlet(chains.BasetenLLMChainlet):
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(gpu=truss_config.Accelerator.H100),
        assets=chains.Assets(secret_keys=["hf_access_token"]),
    )
    llm_config = truss_config.TRTLLMConfiguration(
        build=trt_llm_config.TrussTRTLLMBuildConfiguration(
            base_model=trt_llm_config.TrussTRTLLMModel.LLAMA,
            checkpoint_repository=trt_llm_config.CheckpointRepository(
                source=trt_llm_config.CheckpointSource.HF,
                repo="meta-llama/Llama-3.1-8B-Instruct",
            ),
            max_batch_size=8,
            max_beam_width=1,
            max_seq_len=4096,
            num_builder_gpus=1,
            tensor_parallel_count=1,
        )
    )


@chains.mark_entrypoint
class TestController(chains.ChainletBase):
    def __init__(self, llm=chains.depends(Llama7BChainlet)) -> None:
        self._llm = llm

    async def run_remote(self, prompt: str) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        model_input = baseten_llm.ModelInput(messages=messages)
        async for chunk in self._llm.run_remote(model_input):
            yield chunk
