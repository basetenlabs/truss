from pathlib import Path

from truss.core.patch.dir_signature import directory_content_signature
from truss.core.patch.types import TrussSignature


def calc_truss_signature(truss_dir: Path) -> TrussSignature:
    content_signature = directory_content_signature(truss_dir)
    with (truss_dir / "config.yaml").open("r") as config_file:
        config = config_file.read()
    return TrussSignature(
        content_hashes_by_path=content_signature,
        config=config,
    )
