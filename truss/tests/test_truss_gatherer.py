import os
from pathlib import Path
from typing import List

from truss.base.truss_config import TrussConfig
from truss.truss_handle.build import init_directory
from truss.truss_handle.patch.dir_signature import directory_content_signature
from truss.truss_handle.truss_gatherer import gather
from truss.truss_handle.truss_handle import TrussHandle


def test_gather(custom_model_with_external_package):
    gathered_truss_path = gather(custom_model_with_external_package)
    subdir = gathered_truss_path / "packages" / "subdir"
    sub_module = gathered_truss_path / "packages" / "subdir" / "sub_module.py"
    ext_pkg_top_module = gathered_truss_path / "packages" / "top_module.py"
    ext_pkg_top_module2 = gathered_truss_path / "packages" / "top_module2.py"
    assert subdir.exists()
    assert ext_pkg_top_module.exists()
    assert ext_pkg_top_module2.exists()
    assert sub_module.exists()
    sub_module.unlink()
    subdir.rmdir()
    ext_pkg_top_module.unlink()
    ext_pkg_top_module2.unlink()

    assert _same_dir_content(
        custom_model_with_external_package, gathered_truss_path, ["config.yaml"]
    )


def _truss_with_custom_config(tmp_path: Path, config_path: Path) -> Path:
    """A truss whose external packages are only declared in `config_path`."""
    truss_dir = init_directory(tmp_path / "truss")
    (tmp_path / "ext_pkg").mkdir()
    (tmp_path / "ext_pkg" / "top_module.py").touch()
    config = TrussConfig.from_yaml(truss_dir / "config.yaml")
    config.model_name = "from-custom-config"
    config.external_package_dirs = ["../ext_pkg"]
    config.write_to_yaml_file(config_path)
    return truss_dir


def test_gather_uses_custom_config(tmp_path):
    custom_config = tmp_path / "custom.yaml"
    truss_dir = _truss_with_custom_config(tmp_path, custom_config)

    gathered_path = TrussHandle(truss_dir, config_path=custom_config).gather()

    assert (gathered_path / "packages" / "top_module.py").exists()
    gathered_config = TrussConfig.from_yaml(gathered_path / "config.yaml")
    assert gathered_config.model_name == "from-custom-config"
    assert gathered_config.external_package_dirs == []


def test_gather_does_not_need_default_config(tmp_path):
    custom_config = tmp_path / "custom.yaml"
    truss_dir = _truss_with_custom_config(tmp_path, custom_config)
    (truss_dir / "config.yaml").unlink()

    gathered_path = TrussHandle(truss_dir, config_path=custom_config).gather()

    gathered_config = TrussConfig.from_yaml(gathered_path / "config.yaml")
    assert gathered_config.model_name == "from-custom-config"


def test_gather_refreshes_when_custom_config_changes(tmp_path):
    custom_config = tmp_path / "custom.yaml"
    truss_dir = _truss_with_custom_config(tmp_path, custom_config)
    first_path = TrussHandle(truss_dir, config_path=custom_config).gather()

    config = TrussConfig.from_yaml(custom_config)
    config.model_name = "changed"
    config.write_to_yaml_file(custom_config)
    newer = custom_config.stat().st_mtime + 10
    os.utime(custom_config, (newer, newer))

    second_path = TrussHandle(truss_dir, config_path=custom_config).gather()

    assert second_path == first_path
    gathered_config = TrussConfig.from_yaml(second_path / "config.yaml")
    assert gathered_config.model_name == "changed"


def test_gather_keeps_custom_configs_separate(tmp_path):
    custom_config = tmp_path / "custom.yaml"
    truss_dir = _truss_with_custom_config(tmp_path, custom_config)
    config = TrussConfig.from_yaml(custom_config)
    config.model_name = "other"
    other_config = tmp_path / "other.yaml"
    config.write_to_yaml_file(other_config)

    custom_path = TrussHandle(truss_dir, config_path=custom_config).gather()
    other_path = TrussHandle(truss_dir, config_path=other_config).gather()

    assert custom_path != other_path
    assert TrussConfig.from_yaml(other_path / "config.yaml").model_name == "other"
    assert (
        TrussConfig.from_yaml(custom_path / "config.yaml").model_name
        == "from-custom-config"
    )


def _same_dir_content(dir1: Path, dir2: Path, ignore_paths: List[str]) -> bool:
    sig1 = directory_content_signature(dir1)
    for path in ignore_paths:
        del sig1[path]
    sig2 = directory_content_signature(dir2)
    for path in ignore_paths:
        del sig2[path]
    return sig1 == sig2
