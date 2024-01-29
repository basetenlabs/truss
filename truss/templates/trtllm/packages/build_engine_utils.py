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


def build_engine(
    model_repo: str,
    config: BuildConfig,
    dst: Path,
    hf_auth_token: str,
    tensor_parallelism: int,
    pipeline_parallelism: int,
):
    logging.info(f"building {model_repo} with {config} at {dst}")

    with tempfile.TemporaryDirectory() as model_dst:
        logging.info(f"download model {model_repo}")
        snapshot_download(
            model_repo,
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
