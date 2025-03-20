from typing import AsyncIterator

import truss_chains as chains
from truss.base import trt_llm_config, truss_config


class Llama7BChainlet(chains.EngineBuilderLLMChainlet):
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(gpu=truss_config.Accelerator.H100),
        assets=chains.Assets(secret_keys=["hf_access_token"]),
        options=chains.ChainletOptions(metadata={"tags": ["openai-compatible"]}),
    )
    engine_builder_config = truss_config.TRTLLMConfiguration(
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
    """Example how the engine builder Chainlet can be used by another Chainlet."""

    def __init__(self, llm=chains.depends(Llama7BChainlet)) -> None:
        self._llm = llm

    async def run_remote(self, prompt: str) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        llm_input = chains.EngineBuilderLLMInput(messages=messages)
        async for chunk in self._llm.run_remote(llm_input):
            yield chunk
