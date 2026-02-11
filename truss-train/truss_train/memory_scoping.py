"""Memory scoping for SFT: auto-select H100, H200, or multinode H200 (2x8)."""

import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from truss.base import truss_config

from truss_train.definitions import MemoryRequirements


# GPU memory (GB): H100 80GB, H200 141GB. GPUs per node typically 8.
H100_MEMORY_GB = 80
H200_MEMORY_GB = 141
GPUS_PER_NODE = 8


def _infer_model_params_b(model: str) -> Optional[float]:
    """Infer model size in billions from model name (e.g. Llama-2-7b -> 7, Qwen3-4B -> 4, Llama-235-B -> 235)."""
    name = model.split("/")[-1] if "/" in model else model
    # Match patterns like 7b, 7B, 70b, 4B, 1.5B, 235-B, 235B
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-]?\s*[bB]", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def estimate_memory_gb(
    model_params_b: float,
    per_device_batch_size: int = 2,
    max_seq_length: int = 2048,
) -> float:
    """Estimate GPU memory (GB) for LoRA SFT. Conservative formula."""
    # Base: model weights (bf16) + LoRA + optimizer states
    base_gb = 4 * model_params_b
    # Activations: batch * seq * sqrt(params) factor
    activation_gb = 2 * per_device_batch_size * (max_seq_length / 1024) * (model_params_b**0.5)
    return base_gb + activation_gb + 4  # +4 GB buffer


@dataclass
class ScopedCompute:
    """Result of memory scoping: accelerator spec and node count."""

    accelerator: truss_config.AcceleratorSpec
    node_count: int


def scope_compute(
    memory_gb: float,
) -> ScopedCompute:
    """
    Scope compute to H100, H200, or multinode H200 (2x8).
    Single node: H100 (8x80=640GB) or H200 (8x141=1128GB).
    Multinode: H200 when memory exceeds single-node capacity.
    """
    h100_per_node_gb = GPUS_PER_NODE * H100_MEMORY_GB  # 640
    h200_per_node_gb = GPUS_PER_NODE * H200_MEMORY_GB  # 1128

    if memory_gb <= h100_per_node_gb * 0.8:  # fit on 1 node H100
        return ScopedCompute(
            accelerator=truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator.H100, count=GPUS_PER_NODE
            ),
            node_count=1,
        )
    if memory_gb <= h200_per_node_gb * 0.8:  # fit on 1 node H200
        return ScopedCompute(
            accelerator=truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator.H200, count=GPUS_PER_NODE
            ),
            node_count=1,
        )
    # Multinode H200 (2x8)
    nodes = max(2, math.ceil(memory_gb / (h200_per_node_gb * 0.8)))
    return ScopedCompute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H200, count=GPUS_PER_NODE
        ),
        node_count=nodes,
    )


def scope_from_config(
    model: str,
    memory: Optional[MemoryRequirements],
    model_params_b_override: Optional[float] = None,
) -> Optional[Tuple[truss_config.AcceleratorSpec, int]]:
    """
    If memory config or model_params_b is set, compute (accelerator, node_count).
    model_params_b_override (top-level config) takes precedence over memory.model_params_b.
    Returns None if neither memory nor model_params_b_override is set.
    """
    # Need at least one of: memory block or top-level model_params_b
    if memory is None and model_params_b_override is None:
        return None

    params_b = model_params_b_override or (
        memory.model_params_b if memory else None
    )
    if params_b is None:
        params_b = _infer_model_params_b(model)
    if params_b is None:
        raise ValueError(
            f"Cannot infer model size from '{model}'. Set model_params_b or memory.model_params_b explicitly."
        )

    batch_size = memory.per_device_batch_size if memory else 2
    max_seq = memory.max_seq_length if memory else 2048

    mem_gb = estimate_memory_gb(
        model_params_b=params_b,
        per_device_batch_size=batch_size,
        max_seq_length=max_seq,
    )
    scoped = scope_compute(mem_gb)
    return scoped.accelerator, scoped.node_count
