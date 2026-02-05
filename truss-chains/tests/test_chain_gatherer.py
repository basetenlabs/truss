import pathlib
import tempfile

import pytest
import yaml

from truss_chains.deployment.chain_gatherer import BUNDLED_PACKAGES_DIR, gather_chain


@pytest.fixture
def temp_chain_root():
    """Create a temporary chain root with some files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_root = pathlib.Path(tmpdir) / "chain_root"
        chain_root.mkdir()

        # Create some chainlet files
        (chain_root / "chainlet_a.py").write_text("# Chainlet A")
        (chain_root / "chainlet_b.py").write_text("# Chainlet B")

        # Create a subdirectory with files
        subdir = chain_root / "utils"
        subdir.mkdir()
        (subdir / "helpers.py").write_text("# Helpers")

        yield chain_root


@pytest.fixture
def temp_external_packages():
    """Create temporary external package directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir) / "external"
        base.mkdir()

        # Create first external package
        pkg1 = base / "shared_lib"
        pkg1.mkdir()
        (pkg1 / "__init__.py").write_text("# shared_lib init")
        (pkg1 / "utils.py").write_text("# shared_lib utils")

        # Create second external package
        pkg2 = base / "common_utils"
        pkg2.mkdir()
        (pkg2 / "__init__.py").write_text("# common_utils init")
        (pkg2 / "helpers.py").write_text("# common_utils helpers")

        yield [pkg1, pkg2]


def test_single_external_package(temp_chain_root, temp_external_packages):
    """Single external package should be bundled correctly."""
    ext_pkg = temp_external_packages[0]  # shared_lib
    result = gather_chain(temp_chain_root, [ext_pkg])

    # Result should be a different directory
    assert result != temp_chain_root

    # Original chain contents should be present
    assert (result / "chainlet_a.py").exists()
    assert (result / "chainlet_b.py").exists()
    assert (result / "utils" / "helpers.py").exists()

    # External package contents should be in packages/
    packages_dir = result / BUNDLED_PACKAGES_DIR
    assert packages_dir.exists()

    # Contents of shared_lib should be copied (not the directory itself)
    assert (packages_dir / "__init__.py").exists()
    assert (packages_dir / "utils.py").exists()
    assert (packages_dir / "__init__.py").read_text() == "# shared_lib init"


def test_multiple_external_packages(temp_chain_root, temp_external_packages):
    """Multiple external packages should all be bundled."""
    result = gather_chain(temp_chain_root, temp_external_packages)

    packages_dir = result / BUNDLED_PACKAGES_DIR
    assert packages_dir.exists()

    # Both packages' contents should be present
    # The second __init__.py will overwrite the first (following truss pattern)
    assert (packages_dir / "__init__.py").exists()

    # utils.py from shared_lib and helpers.py from common_utils
    assert (packages_dir / "utils.py").exists()
    assert (packages_dir / "helpers.py").exists()


def test_non_directory_raises_error(temp_chain_root):
    """Non-directory external package should raise an error."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fake_file = pathlib.Path(f.name)

    try:
        with pytest.raises(ValueError, match="is not a directory"):
            gather_chain(temp_chain_root, [fake_file])
    finally:
        fake_file.unlink()


def test_preserves_directory_structure(temp_chain_root):
    """Nested directories in external packages should be preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_pkg = pathlib.Path(tmpdir) / "ext_pkg"
        ext_pkg.mkdir()

        # Create nested structure
        (ext_pkg / "__init__.py").write_text("")
        nested = ext_pkg / "nested"
        nested.mkdir()
        (nested / "__init__.py").write_text("")
        (nested / "deep_module.py").write_text("# deep module")

        result = gather_chain(temp_chain_root, [ext_pkg])

        packages_dir = result / BUNDLED_PACKAGES_DIR
        # The nested directory should be under packages/ (not ext_pkg)
        assert (packages_dir / "nested" / "__init__.py").exists()
        assert (packages_dir / "nested" / "deep_module.py").exists()


def test_clears_external_package_dirs_from_configs(temp_external_packages):
    """external_package_dirs should be cleared from chainlet configs after gathering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_root = pathlib.Path(tmpdir) / "chain_root"
        chain_root.mkdir()

        # Create chainlet directories with config.yaml files
        chainlet_a = chain_root / "chainlet_HelloWorld"
        chainlet_a.mkdir()
        config_a = {
            "model_name": "HelloWorld",
            "external_package_dirs": ["/some/external/path"],
            "requirements": [],
        }
        (chainlet_a / "config.yaml").write_text(yaml.safe_dump(config_a))

        chainlet_b = chain_root / "chainlet_RandInt"
        chainlet_b.mkdir()
        config_b = {
            "model_name": "RandInt",
            "external_package_dirs": [],  # Already empty
            "requirements": [],
        }
        (chainlet_b / "config.yaml").write_text(yaml.safe_dump(config_b))

        # Gather with external packages
        ext_pkg = temp_external_packages[0]
        result = gather_chain(chain_root, [ext_pkg])

        # Check that external_package_dirs is cleared in gathered configs
        gathered_config_a = yaml.safe_load(
            (result / "chainlet_HelloWorld" / "config.yaml").read_text()
        )
        assert gathered_config_a["external_package_dirs"] == []

        gathered_config_b = yaml.safe_load(
            (result / "chainlet_RandInt" / "config.yaml").read_text()
        )
        assert gathered_config_b["external_package_dirs"] == []

        # Other config fields should be preserved
        assert gathered_config_a["model_name"] == "HelloWorld"
        assert gathered_config_b["model_name"] == "RandInt"


def test_packages_directory_bundled_correctly(temp_chain_root):
    """Test that a directory named 'packages' is correctly bundled.

    This verifies the standard convention where external_package_dirs is named
    'packages', ensuring contents are correctly placed in the bundled packages dir.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a 'packages' directory (the standard naming convention)
        packages_ext = pathlib.Path(tmpdir) / "packages"
        packages_ext.mkdir()

        # Create a Python package inside
        my_pkg = packages_ext / "my_package"
        my_pkg.mkdir()
        (my_pkg / "__init__.py").write_text("# my_package init")
        (my_pkg / "module.py").write_text("def foo(): return 'bar'")

        result = gather_chain(temp_chain_root, [packages_ext])

        # The contents of 'packages' should be in the bundled packages dir
        bundled_packages = result / BUNDLED_PACKAGES_DIR
        assert (bundled_packages / "my_package" / "__init__.py").exists()
        assert (bundled_packages / "my_package" / "module.py").exists()
        assert (
            bundled_packages / "my_package" / "module.py"
        ).read_text() == "def foo(): return 'bar'"
