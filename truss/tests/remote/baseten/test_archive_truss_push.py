import shutil
import tarfile
from pathlib import Path

import yaml

from truss.base.constants import CONFIG_FILE
from truss.remote.baseten.core import archive_dir
from truss.truss_handle.truss_handle import TrussHandle


def test_archive_dir_injects_config_yaml_override(
    tmp_path: Path, test_data_path: Path
) -> None:
    """Non-default --config must appear as config.yaml in the uploaded tarball."""
    src = test_data_path / "test_basic_truss"
    truss_dir = tmp_path / "truss"
    shutil.copytree(src, truss_dir)
    (truss_dir / "config.yaml").unlink()
    alt_path = truss_dir / "config.alt.yaml"
    with (src / "config.yaml").open() as f:
        alt_config = yaml.safe_load(f)
    alt_config["model_name"] = "from_alternate_config"
    with alt_path.open("w") as f:
        yaml.safe_dump(alt_config, f)

    handle = TrussHandle(truss_dir, config_path=alt_path)
    default_config = (handle.truss_dir / CONFIG_FILE).resolve()
    assert handle.spec.config_path.resolve() != default_config
    expected_bytes = alt_path.read_bytes()
    archive = archive_dir(truss_dir, config_yaml_override=expected_bytes)
    archive.file.seek(0)

    with tarfile.open(fileobj=archive.file, mode="r:") as tar:
        names = tar.getnames()
        assert "config.yaml" in names
        assert "config.alt.yaml" in names
        member = tar.extractfile("config.yaml")
        assert member is not None
        assert member.read() == expected_bytes

    assert yaml.safe_load(expected_bytes)["model_name"] == "from_alternate_config"
