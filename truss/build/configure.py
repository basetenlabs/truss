import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from truss import load
from truss.patch.hash import directory_content_hash
from truss.patch.signature import calc_truss_signature
from truss.patch.truss_dir_patch_applier import TrussDirPatchApplier
from truss.server.control.patch.types import Patch

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


app = typer.Typer()


@app.command()
def configure_truss_for_build(
    truss_dir: str,
    build_context_path: str,
    use_control_server: bool = False,
    patches_path: Optional[str] = None,
    hash_file_path: Optional[str] = None,
    signature_file_path: Optional[str] = None,
):
    tr = load(truss_dir)

    if patches_path is not None:
        logging.info("Applying patches")
        logger = logging.getLogger("patch_applier")
        patch_applier = TrussDirPatchApplier(Path(truss_dir), logger)
        patches = json.loads(Path(patches_path).read_text())
        patch_applier([Patch.from_dict(patch) for patch in patches])

    # Important to do this before making changes to truss, we want
    # to capture hash of original truss.
    if hash_file_path is not None:
        logging.info("Recording truss hash")
        Path(hash_file_path).write_text(directory_content_hash(Path(truss_dir)))

    if signature_file_path is not None:
        logging.info("Recording truss signature")
        signature_str = json.dumps(calc_truss_signature(Path(truss_dir)).to_dict())
        Path(signature_file_path).write_text(signature_str)

    tr.live_reload(enable=use_control_server)

    logging.debug("Setting up docker build context for truss")

    # check if we have a hf_secret
    tr.docker_build_setup(
        Path(build_context_path), use_hf_secret="HUGGING_FACE_HUB_TOKEN" in os.environ
    )
    logging.info("docker build context is set up for the truss")


if __name__ == "__main__":
    # parse the things
    app()
