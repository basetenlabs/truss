import json
import logging
import os
import pathlib
import sys
from typing import Optional

import click

from truss.base import constants, trt_llm_config
from truss.patch.hash import directory_content_hash
from truss.patch.truss_dir_patch_applier import TrussDirPatchApplier
from truss.templates.control.control.helpers.custom_types import Patch
from truss.truss_handle import truss_handle
from truss.truss_handle.patch import signature

# Note: logging is not picked up in logs UI, only prints.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# MacOS does not allow setting /build without disabling SIP, so allow override
# This is useful for local builds, where we want to build in a different directory
if truss_working_dir := os.environ.get("TRUSS_WORKING_DIR"):
    working_dir = pathlib.Path(truss_working_dir)
else:
    working_dir = pathlib.Path("/")

TRUSS_SRC_DIR = working_dir / "build/model_scaffold"
TRUSS_HASH_FILE = working_dir / "scaffold/truss_hash"
TRUSS_SIGNATURE_FILE = working_dir / "scaffold/truss_signature"
TRUSS_BUILD_CONTEXT_DIR = working_dir / "build/context"


def _fill_trt_llm_versions(
    tr: truss_handle.TrussHandle, image_versions: trt_llm_config.ImageVersions
):
    assert tr.spec.config.trt_llm is not None

    if tr.spec.config.trt_llm.inference_stack == "v2":
        print(f"Using Inference Stack v2 image: {image_versions.v2_llm_image}")
        tr.set_base_image(image_versions.v2_llm_image, "/usr/bin/python3")
    elif tr.spec.config.trt_llm.inference_stack == "v1":
        if (
            tr.spec.config.trt_llm.build.base_model
            == trt_llm_config.TrussTRTLLMModel.ENCODER
        ):
            print(f"Using BEI image: {image_versions.bei_image}")
            tr.set_base_image(image_versions.bei_image, "/usr/bin/python3")
        else:
            print(f"Using Briton image: {image_versions.briton_image}")
            tr.set_base_image(
                image_versions.briton_image, constants.TRTLLM_PYTHON_EXECUTABLE
            )


@click.command()
@click.option("--truss_type", required=True)
@click.option("--trt_llm_image_versions_json")
def docker_build_setup(
    truss_type: str, trt_llm_image_versions_json: Optional[str] = None
) -> None:
    """
    Prepares source and asset files in a build directory (build context), on which a
    docker build command can be run.

    This is to be run for remote builds on baseten.
    Local builds use `TrussHandle.build_serving_docker_image`.
    """
    logging.info("Loading truss")
    tr = truss_handle.TrussHandle(TRUSS_SRC_DIR)
    if tr.spec.config.trt_llm is not None and trt_llm_image_versions_json:
        image_versions = trt_llm_config.ImageVersions.model_validate_json(
            trt_llm_image_versions_json
        )
        _fill_trt_llm_versions(tr, image_versions)

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


if __name__ == "__main__":
    docker_build_setup()
