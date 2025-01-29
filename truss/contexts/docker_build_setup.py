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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# working_dir = pathlib.Path("/")
working_dir = pathlib.Path(os.getcwd())

TRUSS_SRC_DIR = working_dir / "build/model_scaffold"
TRUSS_HASH_FILE = working_dir / "scaffold/truss_hash"
TRUSS_SIGNATURE_FILE = working_dir / "build/truss_signature"
TRUSS_BUILD_CONTEXT_DIR = working_dir / "build/context"


@click.command()
@click.option("--truss_type", required=True)
def docker_build_setup(truss_type: str) -> None:
    print("Loading truss")
    tr = truss_handle.TrussHandle(TRUSS_SRC_DIR)
    print("Truss is loaded")

    if patches_dir := os.environ.get("PATCHES_DIR"):
        print("Applying patches")
        logger = logging.getLogger("patch_applier")
        patch_applier = TrussDirPatchApplier(TRUSS_SRC_DIR, logger)
        patches = json.loads(pathlib.Path(patches_dir).read_text())
        patch_applier([Patch.from_dict(patch) for patch in patches])

    # Important to do this before making changes to truss, we want
    # to capture hash of original truss.
    print("Recording truss hash")
    TRUSS_HASH_FILE.write_text(directory_content_hash(TRUSS_SRC_DIR))

    print("Recording truss signature.")
    sign = signature.calc_truss_signature(TRUSS_SRC_DIR)
    TRUSS_SIGNATURE_FILE.write_text(json.dumps(sign.to_dict()))

    print("Setting up docker build context for truss.")
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
