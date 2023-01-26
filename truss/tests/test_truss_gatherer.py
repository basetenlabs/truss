from pathlib import Path
from typing import List

from truss.patch.dir_signature import directory_content_signature
from truss.truss_gatherer import gather


def test_gather(custom_model_with_external_package):
    gathered_truss_path = gather(custom_model_with_external_package)
    subdir = gathered_truss_path / "packages" / "subdir"
    ext_pkg_file = gathered_truss_path / "packages" / "file.py"
    assert subdir.exists()
    assert ext_pkg_file.exists()
    subdir.rmdir()
    ext_pkg_file.unlink()
    assert _same_dir_content(
        custom_model_with_external_package,
        gathered_truss_path,
        ["config.yaml"],
    )


def _same_dir_content(dir1: Path, dir2: Path, ignore_paths: List[str]) -> bool:
    sig1 = directory_content_signature(dir1)
    for path in ignore_paths:
        del sig1[path]
    sig2 = directory_content_signature(dir2)
    for path in ignore_paths:
        del sig2[path]
    return sig1 == sig2
