"""Tests for custom config path functionality.

This module tests the --config option that allows specifying a custom config file
instead of the default config.yaml in the truss directory.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from truss.base.truss_spec import TrussSpec
from truss.truss_handle.build import load
from truss.truss_handle.truss_handle import TrussHandle


@pytest.fixture
def truss_dir_with_multiple_configs(test_data_path: Path):
    """Create a truss directory with multiple config files for testing."""
    source_dir = test_data_path / "test_basic_truss"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        truss_path = tmp_path / "test_truss"
        shutil.copytree(source_dir, truss_path)

        # Read the original config
        original_config_path = truss_path / "config.yaml"
        with open(original_config_path) as f:
            original_config = yaml.safe_load(f)

        # Create an alternate config with different model_name
        alt_config = original_config.copy()
        alt_config["model_name"] = "alternate-model"
        alt_config_path = truss_path / "config.dev.yaml"
        with open(alt_config_path, "w") as f:
            yaml.dump(alt_config, f)

        # Create another alternate config outside the truss directory
        external_config = original_config.copy()
        external_config["model_name"] = "external-model"
        external_config_path = tmp_path / "external_config.yaml"
        with open(external_config_path, "w") as f:
            yaml.dump(external_config, f)

        yield {
            "truss_dir": truss_path,
            "default_config": original_config_path,
            "alt_config": alt_config_path,
            "external_config": external_config_path,
            "original_model_name": original_config.get("model_name"),
        }


class TestTrussSpecCustomConfigPath:
    """Tests for TrussSpec with custom config path."""

    def test_default_config_path_backwards_compatibility(
        self, truss_dir_with_multiple_configs
    ):
        """Test that TrussSpec without config_path uses default config.yaml."""
        data = truss_dir_with_multiple_configs
        spec = TrussSpec(data["truss_dir"])

        # Should load from default config.yaml
        assert spec.config_path == data["default_config"]
        assert spec.config.model_name == data["original_model_name"]

    def test_custom_config_path_in_truss_dir(self, truss_dir_with_multiple_configs):
        """Test that TrussSpec with config_path loads from that file."""
        data = truss_dir_with_multiple_configs
        spec = TrussSpec(data["truss_dir"], config_path=data["alt_config"])

        # Should load from alternate config
        assert spec.config_path == data["alt_config"]
        assert spec.config.model_name == "alternate-model"

    def test_custom_config_path_external(self, truss_dir_with_multiple_configs):
        """Test that TrussSpec with external config_path works."""
        data = truss_dir_with_multiple_configs
        spec = TrussSpec(data["truss_dir"], config_path=data["external_config"])

        # Should load from external config
        assert spec.config_path == data["external_config"]
        assert spec.config.model_name == "external-model"


class TestTrussHandleCustomConfigPath:
    """Tests for TrussHandle with custom config path."""

    def test_default_config_path_backwards_compatibility(
        self, truss_dir_with_multiple_configs
    ):
        """Test that TrussHandle without config_path uses default config.yaml."""
        data = truss_dir_with_multiple_configs
        handle = TrussHandle(data["truss_dir"])

        # Should load from default config.yaml
        assert handle.spec.config_path == data["default_config"]
        assert handle.spec.config.model_name == data["original_model_name"]

    def test_custom_config_path(self, truss_dir_with_multiple_configs):
        """Test that TrussHandle with config_path loads from that file."""
        data = truss_dir_with_multiple_configs
        handle = TrussHandle(data["truss_dir"], config_path=data["alt_config"])

        # Should load from alternate config
        assert handle.spec.config_path == data["alt_config"]
        assert handle.spec.config.model_name == "alternate-model"

    def test_update_config_preserves_custom_config_path(
        self, truss_dir_with_multiple_configs
    ):
        """Test that _update_config preserves the custom config path after reload."""
        data = truss_dir_with_multiple_configs
        handle = TrussHandle(data["truss_dir"], config_path=data["alt_config"])

        # Update the config (this triggers a reload)
        handle._update_config(description="Test description")

        # After reload, should still use the custom config path
        assert handle.spec.config_path == data["alt_config"]
        assert handle.spec.config.model_name == "alternate-model"
        assert handle.spec.config.description == "Test description"


class TestLoadFunctionCustomConfigPath:
    """Tests for the load() function with custom config path."""

    def test_load_default_config_path_backwards_compatibility(
        self, truss_dir_with_multiple_configs
    ):
        """Test that load() without config_path uses default config.yaml."""
        data = truss_dir_with_multiple_configs
        handle = load(data["truss_dir"])

        # Should load from default config.yaml
        assert handle.spec.config_path == data["default_config"]
        assert handle.spec.config.model_name == data["original_model_name"]

    def test_load_with_custom_config_path(self, truss_dir_with_multiple_configs):
        """Test that load() with config_path loads from that file."""
        data = truss_dir_with_multiple_configs
        handle = load(data["truss_dir"], config_path=data["alt_config"])

        # Should load from alternate config
        assert handle.spec.config_path == data["alt_config"]
        assert handle.spec.config.model_name == "alternate-model"

    def test_load_with_external_config_path(self, truss_dir_with_multiple_configs):
        """Test that load() with external config_path works."""
        data = truss_dir_with_multiple_configs
        handle = load(data["truss_dir"], config_path=data["external_config"])

        # Should load from external config
        assert handle.spec.config_path == data["external_config"]
        assert handle.spec.config.model_name == "external-model"
