from pathlib import Path
from typing import List, Optional

from truss.base.constants import CONFIG_FILE
from truss.base.truss_config import TrussConfig
from truss.truss_handle.patch.custom_types import TrussSignature
from truss.truss_handle.patch.dir_signature import directory_content_signature


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
