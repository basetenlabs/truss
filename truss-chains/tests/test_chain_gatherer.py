import pathlib
import tempfile

import pytest

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


class TestGatherChain:
    def test_no_external_packages_returns_original(self, temp_chain_root):
        """When no external packages, should return original chain_root."""
        result = gather_chain(temp_chain_root, [])
        assert result == temp_chain_root

    def test_single_external_package(self, temp_chain_root, temp_external_packages):
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

    def test_multiple_external_packages(self, temp_chain_root, temp_external_packages):
        """Multiple external packages should all be bundled."""
        result = gather_chain(temp_chain_root, temp_external_packages)

        packages_dir = result / BUNDLED_PACKAGES_DIR
        assert packages_dir.exists()

        # Both packages' contents should be present
        # Since both have __init__.py, one should have a conflict suffix
        init_files = list(packages_dir.glob("__init__*.py"))
        assert len(init_files) == 2

        # utils.py from shared_lib and helpers.py from common_utils
        assert (packages_dir / "utils.py").exists()
        assert (packages_dir / "helpers.py").exists()

    def test_external_package_inside_chain_root_skipped(self, temp_chain_root):
        """External packages already inside chain_root should be skipped."""
        # Create a "external" dir inside chain_root
        internal_ext = temp_chain_root / "internal_ext"
        internal_ext.mkdir()
        (internal_ext / "module.py").write_text("# internal module")

        result = gather_chain(temp_chain_root, [internal_ext])

        # Should return original since ext is inside chain_root
        assert result == temp_chain_root

    def test_deduplication(self, temp_chain_root, temp_external_packages):
        """Same external package path should only be bundled once."""
        ext_pkg = temp_external_packages[0]
        # Pass same path twice
        result = gather_chain(temp_chain_root, [ext_pkg, ext_pkg])

        packages_dir = result / BUNDLED_PACKAGES_DIR
        # Should only have one copy of each file
        init_files = list(packages_dir.glob("__init__*.py"))
        assert len(init_files) == 1

    def test_non_directory_raises_error(self, temp_chain_root):
        """Non-directory external package should raise an error."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            fake_file = pathlib.Path(f.name)

        try:
            with pytest.raises(ValueError, match="is not a directory"):
                gather_chain(temp_chain_root, [fake_file])
        finally:
            fake_file.unlink()

    def test_name_conflict_handling(self, temp_chain_root):
        """Files with same name from different packages should get unique names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = pathlib.Path(tmpdir)

            # Create two packages with same file name
            pkg1 = base / "pkg1"
            pkg1.mkdir()
            (pkg1 / "config.py").write_text("# config from pkg1")

            pkg2 = base / "pkg2"
            pkg2.mkdir()
            (pkg2 / "config.py").write_text("# config from pkg2")

            result = gather_chain(temp_chain_root, [pkg1, pkg2])

            packages_dir = result / BUNDLED_PACKAGES_DIR
            config_files = list(packages_dir.glob("config*.py"))

            # Should have two config files with different names
            assert len(config_files) == 2
            contents = {f.read_text() for f in config_files}
            assert "# config from pkg1" in contents
            assert "# config from pkg2" in contents

    def test_preserves_directory_structure(self, temp_chain_root):
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
