from pathlib import Path

from schema import EngineBuildArgs


def build_engine_from_config_args(
    engine_build_args: EngineBuildArgs,
    dst: Path,
):
    import sys

    sys.path.append("/app/baseten")

    import os
    import shutil

    # NOTE: These are provided by the underlying base image
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
