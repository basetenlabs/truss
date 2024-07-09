import json 

from pathlib import Path
from typing import List, Optional

from truss.constants import CONFIG_FILE
from truss.patch.dir_signature import directory_content_signature
from truss.patch.types import TrussSignature
from truss.truss_config import TrussConfig



def calc_truss_signature(
    truss_dir: Path, ignore_patterns: Optional[List[str]] = None
) -> TrussSignature:
    content_signature = directory_content_signature(truss_dir, ignore_patterns)
    config = TrussConfig.to_signature_config(truss_dir / CONFIG_FILE)
    return TrussSignature(
        content_hashes_by_path=content_signature,
        config=config,
    )
