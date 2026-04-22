from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


def _format_cli_arg(key: str, value: Any) -> str:
    """Convert a snake_case key to a kebab-case CLI flag."""
    flag = key.replace("_", "-")
    if isinstance(value, bool):
        return f"--{flag}" if value else ""
    return f"--{flag} {value}"


class VLLMConfiguration(BaseModel):
    """Configuration for serving models with the vLLM inference engine.

    When this block is present in config.yaml, Truss will automatically:
    1. Use the vLLM OpenAI Docker image as the base image.
    2. Construct a ``vllm serve`` start command from these settings.
    3. Deploy via the docker_server path (with nginx proxy).

    Most fields map directly to ``vllm serve`` CLI arguments.  Additional
    arguments can be passed via ``extra_args``.
    """

    model: str = Field(
        ...,
        description="Model ID or local path to serve. e.g. meta-llama/Llama-2-7b-hf",
    )
    port: int = Field(
        default=8000, description="Port for the vLLM OpenAI-compatible server."
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the vLLM server.")
    tensor_parallel_size: Optional[int] = Field(
        default=None, description="Number of GPUs for tensor parallelism."
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of GPU memory to use (0.0 - 1.0).",
    )
    max_model_len: Optional[int] = Field(
        default=None, description="Maximum sequence length the model can process."
    )
    dtype: Optional[str] = Field(
        default=None,
        description="Data type for model weights, e.g. auto, half, float16, bfloat16, float32.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization method, e.g. awq, gptq, fp8, squeezellm.",
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code from HuggingFace."
    )
    served_model_name: Optional[str] = Field(
        default=None, description="Name to expose in the OpenAI API."
    )
    extra_args: list[str] = Field(
        default_factory=list, description="Additional CLI arguments for vllm serve."
    )

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _validate_gpu_memory_utilization(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0.0 or v > 1.0):
            raise ValueError("gpu_memory_utilization must be in (0.0, 1.0]")
        return v

    def build_start_command(self) -> str:
        """Construct the ``vllm serve`` CLI command from this configuration."""
        cmd_parts = ["vllm serve", self.model]

        # Map fields directly to CLI flags.
        simple_flags = {
            "port": self.port,
            "host": self.host,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "served_model_name": self.served_model_name,
        }
        for key, value in simple_flags.items():
            if value is not None:
                cmd_parts.append(_format_cli_arg(key, value))

        if self.trust_remote_code:
            cmd_parts.append("--trust-remote-code")

        for arg in self.extra_args:
            cmd_parts.append(arg)

        return " ".join(cmd_parts)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return super().model_dump(**kwargs)
