"""Tests for truss migrate command."""

from truss.base.truss_config import ExternalDataItem, ModelRepo, ModelRepoSourceKind
from truss.cli.migrate_commands import (
    convert_external_data_to_weights,
    convert_model_repo_to_weights,
    generate_mount_location_for_model,
    generate_source_uri,
    migrate_config,
)


class TestGenerateSourceUri:
    """Tests for generate_source_uri function."""

    def test_hf_with_revision(self):
        """HuggingFace repo with revision."""
        model = ModelRepo(
            repo_id="meta-llama/Llama-2-7b",
            revision="main",
            use_volume=True,
            volume_folder="llama",
            kind=ModelRepoSourceKind.HF,
        )
        assert generate_source_uri(model) == "hf://meta-llama/Llama-2-7b@main"

    def test_hf_without_revision(self):
        """HuggingFace repo without revision (v1 style)."""
        model = ModelRepo(
            repo_id="meta-llama/Llama-2-7b",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.HF,
        )
        assert generate_source_uri(model) == "hf://meta-llama/Llama-2-7b"

    def test_gcs_with_prefix(self):
        """GCS with gs:// prefix already."""
        model = ModelRepo(
            repo_id="gs://my-bucket/models",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.GCS,
        )
        assert generate_source_uri(model) == "gs://my-bucket/models"

    def test_gcs_without_prefix(self):
        """GCS without prefix."""
        model = ModelRepo(
            repo_id="my-bucket/models",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.GCS,
        )
        assert generate_source_uri(model) == "gs://my-bucket/models"

    def test_s3_with_prefix(self):
        """S3 with s3:// prefix already."""
        model = ModelRepo(
            repo_id="s3://my-bucket/models",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.S3,
        )
        assert generate_source_uri(model) == "s3://my-bucket/models"

    def test_azure_with_prefix(self):
        """Azure with azure:// prefix already."""
        model = ModelRepo(
            repo_id="azure://account/container",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.AZURE,
        )
        assert generate_source_uri(model) == "azure://account/container"


class TestGenerateMountLocation:
    """Tests for generate_mount_location_for_model function."""

    def test_v2_with_volume_folder(self):
        """v2 model_cache with volume_folder."""
        model = ModelRepo(
            repo_id="meta-llama/Llama-2-7b",
            revision="main",
            use_volume=True,
            volume_folder="llama-7b",
            kind=ModelRepoSourceKind.HF,
        )
        assert generate_mount_location_for_model(model) == "/app/model_cache/llama-7b"

    def test_v1_hf_sanitizes_repo_id(self):
        """v1 HuggingFace sanitizes repo_id (/ -> _)."""
        model = ModelRepo(
            repo_id="meta-llama/Llama-2-7b",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.HF,
        )
        assert (
            generate_mount_location_for_model(model)
            == "/app/model_cache/meta-llama_Llama-2-7b"
        )

    def test_v1_gcs_uses_bucket_name(self):
        """v1 GCS uses bucket name."""
        model = ModelRepo(
            repo_id="gs://my-bucket/path/to/model",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.GCS,
        )
        assert generate_mount_location_for_model(model) == "/app/model_cache/my-bucket"

    def test_v1_s3_uses_bucket_name(self):
        """v1 S3 uses bucket name."""
        model = ModelRepo(
            repo_id="s3://my-s3-bucket/path",
            revision="",
            use_volume=False,
            kind=ModelRepoSourceKind.S3,
        )
        assert (
            generate_mount_location_for_model(model) == "/app/model_cache/my-s3-bucket"
        )


class TestConvertModelRepoToWeights:
    """Tests for convert_model_repo_to_weights function."""

    def test_basic_v2_hf(self):
        """Convert v2 HuggingFace model."""
        model = ModelRepo(
            repo_id="meta-llama/Llama-2-7b",
            revision="main",
            use_volume=True,
            volume_folder="llama-7b",
            kind=ModelRepoSourceKind.HF,
        )
        result = convert_model_repo_to_weights(model)
        assert result["source"] == "hf://meta-llama/Llama-2-7b@main"
        assert result["mount_location"] == "/app/model_cache/llama-7b"
        assert result["auth_secret_name"] == "hf_access_token"

    def test_with_patterns(self):
        """Convert model with allow/ignore patterns."""
        model = ModelRepo(
            repo_id="meta-llama/Llama-2-7b",
            revision="main",
            use_volume=True,
            volume_folder="llama-7b",
            kind=ModelRepoSourceKind.HF,
            allow_patterns=["*.safetensors"],
            ignore_patterns=["*.md"],
        )
        result = convert_model_repo_to_weights(model)
        assert result["allow_patterns"] == ["*.safetensors"]
        assert result["ignore_patterns"] == ["*.md"]

    def test_with_custom_secret(self):
        """Convert model with custom secret name."""
        model = ModelRepo(
            repo_id="meta-llama/Llama-2-7b",
            revision="main",
            use_volume=True,
            volume_folder="llama-7b",
            kind=ModelRepoSourceKind.HF,
            runtime_secret_name="my_hf_token",
        )
        result = convert_model_repo_to_weights(model)
        assert result["auth_secret_name"] == "my_hf_token"


class TestConvertExternalDataToWeights:
    """Tests for convert_external_data_to_weights function."""

    def test_basic_external_data(self):
        """Convert basic external_data item."""
        item = ExternalDataItem(
            url="https://example.com/models/weights.bin",
            local_data_path="weights/model.bin",
        )
        result = convert_external_data_to_weights(item)
        assert result["source"] == "https://example.com/models/weights.bin"
        assert result["mount_location"] == "/app/data/weights/model.bin"

    def test_nested_path(self):
        """Convert external_data with nested path."""
        item = ExternalDataItem(
            url="https://cdn.example.com/deep/nested/path/model.safetensors",
            local_data_path="models/base/adapter.safetensors",
        )
        result = convert_external_data_to_weights(item)
        assert result["mount_location"] == "/app/data/models/base/adapter.safetensors"


class TestMigrateConfig:
    """Tests for migrate_config function."""

    def test_migrate_v2_model_cache(self):
        """Migrate v2 model_cache to weights."""
        config = {
            "model_name": "test",
            "model_cache": [
                {
                    "repo_id": "meta-llama/Llama-2-7b",
                    "revision": "main",
                    "use_volume": True,
                    "volume_folder": "llama-7b",
                }
            ],
        }
        migrated, warnings = migrate_config(config)

        assert "model_cache" not in migrated
        assert "weights" in migrated
        assert len(migrated["weights"]) == 1
        assert migrated["weights"][0]["source"] == "hf://meta-llama/Llama-2-7b@main"
        assert migrated["weights"][0]["mount_location"] == "/app/model_cache/llama-7b"
        assert len(warnings) == 0

    def test_migrate_v1_model_cache_warns(self):
        """Migrate v1 model_cache warns about model.py changes."""
        config = {
            "model_name": "test",
            "model_cache": [{"repo_id": "meta-llama/Llama-2-7b", "use_volume": False}],
        }
        migrated, warnings = migrate_config(config)

        assert "model_cache" not in migrated
        assert "weights" in migrated
        assert len(warnings) == 1
        assert "model.py" in warnings[0]

    def test_migrate_external_data(self):
        """Migrate external_data to weights."""
        config = {
            "model_name": "test",
            "external_data": [
                {
                    "url": "https://example.com/weights.bin",
                    "local_data_path": "weights.bin",
                }
            ],
        }
        migrated, warnings = migrate_config(config)

        assert "external_data" not in migrated
        assert "weights" in migrated
        assert len(migrated["weights"]) == 1
        assert migrated["weights"][0]["source"] == "https://example.com/weights.bin"
        assert migrated["weights"][0]["mount_location"] == "/app/data/weights.bin"

    def test_migrate_combined(self):
        """Migrate both model_cache and external_data."""
        config = {
            "model_name": "test",
            "model_cache": [
                {
                    "repo_id": "meta-llama/Llama-2-7b",
                    "revision": "main",
                    "use_volume": True,
                    "volume_folder": "llama-7b",
                }
            ],
            "external_data": [
                {
                    "url": "https://example.com/adapter.bin",
                    "local_data_path": "adapter.bin",
                }
            ],
        }
        migrated, warnings = migrate_config(config)

        assert "model_cache" not in migrated
        assert "external_data" not in migrated
        assert "weights" in migrated
        assert len(migrated["weights"]) == 2

    def test_migrate_empty_config(self):
        """Empty config returns same config."""
        config = {"model_name": "test"}
        migrated, warnings = migrate_config(config)

        assert migrated == config
        assert len(warnings) == 0

    def test_preserves_other_fields(self):
        """Migration preserves other config fields."""
        config = {
            "model_name": "test",
            "python_version": "py311",
            "requirements": ["torch", "transformers"],
            "model_cache": [
                {
                    "repo_id": "test/model",
                    "revision": "main",
                    "use_volume": True,
                    "volume_folder": "model",
                }
            ],
        }
        migrated, _ = migrate_config(config)

        assert migrated["model_name"] == "test"
        assert migrated["python_version"] == "py311"
        assert migrated["requirements"] == ["torch", "transformers"]
