from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

# todo(Abu): consolidate this with build_engine.py
# Enums / BaseClass is duplicated from build_engine so that we know what args / configurations
# are available to us at run-time.


class Quant(Enum):
    NO_QUANT = "no_quant"
    WEIGHTS_ONLY = "weights_only"
    WEIGHTS_KV_INT8 = "weights_kv_int8"
    SMOOTH_QUANT = "smooth_quant"


class EngineType(Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"


class ArgsConfig(BaseModel):
    max_input_len: Optional[int] = None
    max_output_len: Optional[int] = None
    max_batch_size: Optional[int] = None
    tp_size: Optional[int] = None
    pp_size: Optional[int] = None
    world_size: Optional[int] = None
    gather_all_token_logits: Optional[bool] = None
    multi_block_mode: Optional[bool] = None
    remove_input_padding: Optional[bool] = None
    use_gpt_attention_plugin: Optional[str] = None
    paged_kv_cache: Optional[bool] = None
    use_inflight_batching: Optional[bool] = None
    enable_context_fmha: Optional[bool] = None
    use_gemm_plugin: Optional[str] = None
    use_weight_only: Optional[bool] = None
    output_dir: Optional[str] = None
    model_dir: Optional[str] = None
    ft_model_dir: Optional[str] = None
    dtype: Optional[str] = None
    int8_kv_cache: Optional[bool] = None
    use_smooth_quant: Optional[bool] = None
    per_token: Optional[bool] = None
    per_channel: Optional[bool] = None
    parallel_build: Optional[bool] = None

    # to disable warning because `model_dir` starts with `model_` prefix
    model_config = ConfigDict(protected_namespaces=())

    def as_command_arguments(self):
        non_bool_args = [
            element
            for arg, value in self.dict().items()
            for element in [f"--{arg}", str(value)]
            if value is not None and not isinstance(value, bool)
        ]
        bool_args = [
            f"--{arg}"
            for arg, value in self.dict().items()
            if isinstance(value, bool) and value
        ]
        return non_bool_args + bool_args


class CalibrationConfig(BaseModel):
    kv_cache: Optional[bool] = None  # either to calibrate kv cache
    sq_alpha: Optional[float] = None

    def cache_path(self) -> Path:
        if self.kv_cache is not None:
            return Path("kv_cache")
        else:
            return Path(f"sq_{self.sq_alpha}")


class EngineBuildArgs(BaseModel):
    repo: Optional[str] = None
    args: Optional[ArgsConfig] = None
    quant: Optional[Quant] = None
    calibration: Optional[CalibrationConfig] = None
    engine_type: Optional[EngineType] = None

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            repo=config["repo"] if "repo" in config else None,
            args=ArgsConfig(**config["args"]),
            quant=Quant(config["quant"] if "quant" in config else None),
            calibration=CalibrationConfig(
                **config["calibration"] if "calibration" in config else None
            ),
            engine_type=EngineType(
                config["engine_type"] if "engine_type" in config else None
            ),
        )


def build_engine_from_config_args(
    engine_build_args: EngineBuildArgs,
    dst: Path,
):
    import sys

    sys.path.append("/app/baseten")

    import os
    import shutil

    from build_engine import Engine, build_engine
    from trtllm_utils import docker_tag_aware_file_cache

    engine = Engine(**engine_build_args.model_dump())

    with docker_tag_aware_file_cache("/root/.cache/trtllm"):
        built_engine = build_engine(engine, download_remote=True)

        if not os.path.exists(dst):
            os.makedirs(dst)

        for filename in os.listdir(str(built_engine)):
            source_file = os.path.join(str(built_engine), filename)
            destination_file = os.path.join(dst, filename)
            if not os.path.exists(destination_file):
                shutil.copy(source_file, destination_file)

        return dst
