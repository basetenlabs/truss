from pathlib import Path
from typing import List, Optional

from truss.patch.dir_signature import directory_content_signature
from truss.patch.types import TrussSignature


def calc_truss_signature(
    truss_dir: Path, ignore_patterns: Optional[List[str]] = None
) -> TrussSignature:
    content_signature = directory_content_signature(truss_dir, ignore_patterns)
    with (truss_dir / "config.yaml").open("r") as config_file:
        config = config_file.read()
    return TrussSignature(
        content_hashes_by_path=content_signature,
        config=config,
    )
