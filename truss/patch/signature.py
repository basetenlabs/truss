from pathlib import Path
from typing import List, Optional

from truss.constants import CONFIG_FILE
from truss.patch.custom_types import TrussSignature
from truss.patch.dir_signature import directory_content_signature
from truss.truss_config import TrussConfig


def calc_truss_signature(
    truss_dir: Path, ignore_patterns: Optional[List[str]] = None
) -> TrussSignature:
    content_signature = directory_content_signature(truss_dir, ignore_patterns)
    config_path = truss_dir / CONFIG_FILE
    with (config_path).open("r") as config_file:
        config = config_file.read()
    requirements = TrussConfig.load_requirements_file_from_filepath(config_path)
    return TrussSignature(
        content_hashes_by_path=content_signature,
        config=config,
        requirements_file_requirements=requirements,
    )
