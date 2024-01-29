import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from pydantic import BaseModel


class BuildConfig(BaseModel):
    cmd: str
    args: str

    output_arg: str = "--output_dir"
    model_arg: str = "--model_dir"
    tensor_parallelism_arg: str = "--tp_size"
    pipeline_parallelism_arg: str = "--pp_size"
    world_arg: str = "--world_size"


def build_engine_from_cmd_args(
    hf_model_repo: str,
    config: BuildConfig,
    dst: Path,
    hf_auth_token: str,
    tensor_parallelism: int,
    pipeline_parallelism: int,
):
    """This implementation directly runs a user-provided command to build the engine."""
    logging.info(f"building {hf_model_repo} with {config} at {dst}")

    with tempfile.TemporaryDirectory() as model_dst:
        logging.info(f"download model {hf_model_repo}")
        snapshot_download(
            hf_model_repo,
            local_dir=model_dst,
            max_workers=4,
            **({"use_auth_token": hf_auth_token} if hf_auth_token is not None else {}),
        )
        build_cmd = (
            [sys.executable, "/app/tensorrt_llm/" + config.cmd]
            + config.args.split(" ")
            + [config.output_arg, str(dst)]
            + [config.model_arg, str(model_dst)]
            + [config.tensor_parallelism_arg, str(tensor_parallelism)]
            + [config.pipeline_parallelism_arg, str(pipeline_parallelism)]
            + [config.world_arg, str(tensor_parallelism * pipeline_parallelism)]
        )
        logging.info(f"build engine with command \"{' '.join(build_cmd)}\"")
        completed_process = subprocess.run(build_cmd, capture_output=False)

        if completed_process.returncode != 0:
            raise Exception(
                f"build failed with {completed_process.returncode} exit code"
            )


def build_engine_from_config_args(
    engine_args: dict,
    hf_model_repository: str,
    dst: Path,
):
    import sys

    sys.path.append("/app/baseten")

    import os
    import shutil

    from build_engine import Engine, build_engine
    from trtllm_utils import docker_tag_aware_file_cache

    engine = Engine(**engine_args)
    engine.repo = hf_model_repository

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
