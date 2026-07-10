from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import Field

try:
    from truss.base import custom_types

    PydanticTrTBaseModel = custom_types.ConfigModel
except ImportError:
    from pydantic import BaseModel as PydanticTrTBaseModel


class TrussLLMSharedConfig(PydanticTrTBaseModel):
    model: Optional[str] = Field(
        default=None,
        description="Model ID or local path, e.g. meta-llama/Llama-2-7b-hf. "
        "For trt_llm, this is an alias for build.checkpoint_repository.repo.",
    )
    revision: Optional[str] = Field(
        default=None, description="Model revision / branch."
    )
    dtype: Optional[str] = Field(
        default=None,
        description="Data type, e.g. auto, half, float16, bfloat16, float32.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization method, e.g. fp8, fp8_kv, fp4, awq, gptq, squeezellm.",
    )
    tensor_parallel_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of GPUs for tensor parallelism. Defaults to accelerator.count.",
    )
    max_model_len: Optional[int] = Field(
        default=None, ge=1, description="Maximum sequence length the model can process."
    )
    served_model_name: Optional[str] = Field(
        default=None, description="Name to expose in the OpenAI API."
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code from HuggingFace."
    )
    extra_args: List[str] = Field(
        default_factory=list,
        description="Additional CLI arguments (vLLM) or forwarded args for escape hatch.",
    )
    patch_kwargs: Dict[str, Union[bool, str, int, float, dict, list, None]] = Field(
        default_factory=dict,
        description="Escape hatch for engine-specific kwargs not covered by typed fields. "
        "For vLLM, these are injected as --kebab-case flags. For TRT-LLM v2, they patch the runtime config.",
    )
