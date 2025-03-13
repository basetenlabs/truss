import json
import logging
import os
import pathlib
import sys

import click

from truss.patch.hash import directory_content_hash
from truss.patch.truss_dir_patch_applier import TrussDirPatchApplier
from truss.templates.control.control.helpers.custom_types import Patch
from truss.truss_handle import truss_handle
from truss.truss_handle.patch import signature

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

working_dir = pathlib.Path("/")

TRUSS_SRC_DIR = working_dir / "build/model_scaffold"
TRUSS_HASH_FILE = working_dir / "scaffold/truss_hash"
TRUSS_SIGNATURE_FILE = working_dir / "scaffold/truss_signature"
TRUSS_BUILD_CONTEXT_DIR = working_dir / "build/context"


@click.command()
@click.option("--truss_type", required=True)
def docker_build_setup(truss_type: str) -> None:
    """
    Prepares source and asset files in a build directory (build context), on which a
    docker build command can be run.

    This is to be run for remote builds on baseten.
    Local builds use `TrussHandle.build_serving_docker_image`.
    """
    logging.info("Loading truss")
    tr = truss_handle.TrussHandle(TRUSS_SRC_DIR)
    logging.info("Truss is loaded")

    if patches_dir := os.environ.get("PATCHES_DIR"):
        logging.info("Applying patches")
        logger = logging.getLogger("patch_applier")
        patch_applier = TrussDirPatchApplier(TRUSS_SRC_DIR, logger)
        patches = json.loads(pathlib.Path(patches_dir).read_text())
        patch_applier([Patch.from_dict(patch) for patch in patches])

    # Important to do this before making changes to truss, we want
    # to capture hash of original truss.
    logging.info("Recording truss hash")
    TRUSS_HASH_FILE.write_text(directory_content_hash(TRUSS_SRC_DIR))

    logging.info("Recording truss signature.")
    sign = signature.calc_truss_signature(TRUSS_SRC_DIR)
    TRUSS_SIGNATURE_FILE.write_text(json.dumps(sign.to_dict()))

    if truss_type == "server_control":
        tr.live_reload(enable=True)
    else:
        tr.live_reload(enable=False)

    # check if we have a hf_secret
    tr.docker_build_setup(
        TRUSS_BUILD_CONTEXT_DIR, use_hf_secret="HUGGING_FACE_HUB_TOKEN" in os.environ
    )
    print("Docker build context is set up for the truss.")


if __name__ == "__main__":
    docker_build_setup()
