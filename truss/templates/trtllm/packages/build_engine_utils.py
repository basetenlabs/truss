from pathlib import Path

from schema import EngineBuildArgs


def build_engine_from_config_args(
    engine_build_args: EngineBuildArgs,
    dst: Path,
):
    import os
    import shutil
    import sys

    # NOTE: These are provided by the underlying base image
    # TODO(Abu): Remove this when we have a better way of handling this
    sys.path.append("/app/baseten")
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
