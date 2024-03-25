from pathlib import Path
from typing import Optional

from builder.types import TrussTRTLLMConfiguration


def build_engine_from_config_args(
    truss_trtllm_configuration: TrussTRTLLMConfiguration,
    dst: Path,
    checkpoint_dir_path: Optional[Path] = None,
):
    # NOTE: These are provided by the underlying base image
    # TODO(Abu): Remove this when we have a better way of handling this
    from builder.main import build_engine

    build_engine(
        engine_configuration=truss_trtllm_configuration,
        engine_serialization_path=dst,
        # If checkpoint_dir_path is provided, we'll look there for the
        # weight files. If not, we will attempt to use the `huggingface_ckpt_repository`
        # key in the `truss_trtllm_configuration` to download the weights.
        checkpoint_dir_path=checkpoint_dir_path,
    )
    return dst
