import pathlib
from unittest import mock

import pytest

from truss.base import truss_config
from truss.remote.baseten import custom_types as b10_types
from truss_train import deployment
from truss_train.definitions import (
    CacheConfig,
    Compute,
    Image,
    Runtime,
    TrainingJob,
    Workspace,
)


@mock.patch("truss.remote.baseten.utils.transfer.multipart_upload_boto3")
@mock.patch("truss.remote.baseten.api.BasetenApi.get_blob_credentials")
@pytest.mark.parametrize("enable_cache", [True, False])
@pytest.mark.parametrize(
    "compute",
    [
        Compute(),
        Compute(
            accelerator=truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator.T4, count=1
            )
        ),
        Compute(
            node_count=2,
            accelerator=truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator.H100, count=8
            ),
        ),
    ],
)
@pytest.mark.parametrize("truss_user_env", [None, "with_env"])
def test_prepare_push(
    get_blob_credentials_mock,
    multipart_upload_boto3_mock,
    compute,
    enable_cache,
    truss_user_env,
):
    if truss_user_env == "with_env":
        truss_user_env = b10_types.TrussUserEnv(
            truss_client_version="1.0.0",
            python_version="3.9.0",
            pydantic_version="2.0.0",
            mypy_version=None,
            git_info=None,
        )
    mock_api = mock.Mock()
    mock_api.get_blob_credentials.return_value = {
        "s3_bucket": "test-s3-bucket",
        "s3_key": "test-s3-key",
        "creds": {},
    }

    prepared_job = deployment.prepare_push(
        mock_api,
        pathlib.Path(__file__).parent,
        TrainingJob(
            image=Image(base_image="hello-world"),
            compute=compute,
            runtime=Runtime(
                cache_config=CacheConfig(
                    enabled=enable_cache, enable_legacy_hf_mount=True
                )
            ),
        ),
        truss_user_env=truss_user_env,
    )
    assert len(prepared_job.runtime_artifacts) == 1
    assert prepared_job.runtime_artifacts[0].s3_key == "test-s3-key"
    assert prepared_job.runtime_artifacts[0].s3_bucket == "test-s3-bucket"
    if compute.accelerator:
        assert (
            prepared_job.compute.accelerator.accelerator
            == compute.accelerator.accelerator
        )
    else:
        assert prepared_job.compute.accelerator is None
    assert prepared_job.runtime.cache_config.enabled == enable_cache
    assert prepared_job.runtime.cache_config.enable_legacy_hf_mount
    assert prepared_job.truss_user_env == truss_user_env
    # ensure that serialization works
    prepared_job.model_dump()


class TestGatherTrainingDir:
    """Tests for _gather_training_dir and workspace functionality."""

    def test_no_workspace_returns_none(self, tmp_path):
        config_path = tmp_path / "config.py"
        config_path.touch()

        result = deployment._gather_training_dir(config_path.parent, workspace=None)
        assert result is None

    def test_empty_workspace_returns_none(self, tmp_path):
        config_path = tmp_path / "config.py"
        config_path.touch()

        workspace = Workspace()
        result = deployment._gather_training_dir(
            config_path.parent, workspace=workspace
        )
        assert result is None

    def test_workspace_root_contains_config(self, tmp_path):
        # Structure:
        # tmp_path/
        #   project/
        #     config.py
        #     train.py
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()
        (project_dir / "train.py").write_text("# training code")

        workspace = Workspace(workspace_root=".")
        result = deployment._gather_training_dir(
            config_path.parent, workspace=workspace
        )

        assert result is not None
        assert (result / "config.py").exists()
        assert (result / "train.py").exists()

    def test_workspace_root_parent_directory(self, tmp_path):
        # Structure:
        # tmp_path/
        #   myproject/
        #     subdir/
        #       config.py
        #     shared.py
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        subdir = project_dir / "subdir"
        subdir.mkdir()
        config_path = subdir / "config.py"
        config_path.touch()
        (project_dir / "shared.py").write_text("# shared code")

        workspace = Workspace(workspace_root="..")
        result = deployment._gather_training_dir(
            config_path.parent, workspace=workspace
        )

        assert result is not None
        assert (result / "subdir" / "config.py").exists()
        assert (result / "shared.py").exists()

    def test_external_dirs_copied(self, tmp_path):
        # Structure:
        # tmp_path/
        #   project/
        #     config.py
        #   shared_lib/
        #     utils.py
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        shared_lib = tmp_path / "shared_lib"
        shared_lib.mkdir()
        (shared_lib / "utils.py").write_text("# utils")

        workspace = Workspace(external_dirs=["../shared_lib"])
        result = deployment._gather_training_dir(
            config_path.parent, workspace=workspace
        )

        assert result is not None
        assert (result / "config.py").exists()
        assert (result / "shared_lib" / "utils.py").exists()

    def test_exclude_dirs_excluded(self, tmp_path):
        # Structure:
        # tmp_path/
        #   project/
        #     config.py
        #     data/
        #       big_file.bin
        #     tests/
        #       test_foo.py
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        data_dir = project_dir / "data"
        data_dir.mkdir()
        (data_dir / "big_file.bin").write_bytes(b"big data")

        tests_dir = project_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_foo.py").write_text("# test")

        workspace = Workspace(exclude_dirs=["./data", "./tests"])
        result = deployment._gather_training_dir(
            config_path.parent, workspace=workspace
        )

        assert result is not None
        assert (result / "config.py").exists()
        assert not (result / "data").exists()
        assert not (result / "tests").exists()

    def test_workspace_with_all_options(self, tmp_path):
        # Structure:
        # tmp_path/
        #   myproject/
        #     subdir/
        #       config.py
        #     src/
        #       main.py
        #     data/
        #       dataset.bin
        #   external/
        #     helpers.py
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        subdir = project_dir / "subdir"
        subdir.mkdir()
        config_path = subdir / "config.py"
        config_path.touch()

        src_dir = project_dir / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("# main")

        data_dir = project_dir / "data"
        data_dir.mkdir()
        (data_dir / "dataset.bin").write_bytes(b"data")

        external_dir = tmp_path / "external"
        external_dir.mkdir()
        (external_dir / "helpers.py").write_text("# helpers")

        workspace = Workspace(
            workspace_root="..",
            external_dirs=["../../external"],
            exclude_dirs=["../data"],
        )
        result = deployment._gather_training_dir(
            config_path.parent, workspace=workspace
        )

        assert result is not None
        assert (result / "subdir" / "config.py").exists()
        assert (result / "src" / "main.py").exists()
        assert not (result / "data").exists()
        assert (result / "external" / "helpers.py").exists()


class TestWorkspaceValidation:
    """Tests for workspace validation errors."""

    def test_config_not_in_workspace_root_fails(self, tmp_path):
        # Structure:
        # tmp_path/
        #   project/
        #     config.py
        #   other_project/
        #     (empty)
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        other_project = tmp_path / "other_project"
        other_project.mkdir()

        workspace = Workspace(workspace_root="../other_project")

        with pytest.raises(ValueError, match="must be inside workspace_root"):
            deployment._gather_training_dir(config_path.parent, workspace=workspace)

    def test_external_dir_inside_workspace_root_warns_and_skips(self, tmp_path, capsys):
        # Structure:
        # tmp_path/
        #   project/
        #     config.py
        #     subdir/
        #       file.py
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        subdir = project_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.py").write_text("# file")

        workspace = Workspace(external_dirs=["./subdir"])

        # Should succeed but skip the external_dir
        result = deployment._gather_training_dir(
            config_path.parent, workspace=workspace
        )

        # The subdir is included via workspace_root, not as separate external dir
        assert result is not None
        assert (result / "config.py").exists()
        assert (result / "subdir" / "file.py").exists()

    def test_external_dir_name_collision_with_workspace_contents_fails(self, tmp_path):
        # Structure:
        # tmp_path/
        #   project/
        #     config.py
        #     src/        <- exists in workspace
        #   other/
        #     src/        <- same name, would collide
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()
        (project_dir / "src").mkdir()

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        collision_dir = other_dir / "src"
        collision_dir.mkdir()

        workspace = Workspace(external_dirs=["../other/src"])

        with pytest.raises(ValueError, match="Name collision"):
            deployment._gather_training_dir(config_path.parent, workspace=workspace)

    def test_external_dir_name_collision_between_external_dirs_fails(self, tmp_path):
        # Structure:
        # tmp_path/
        #   project/
        #     config.py
        #   dir_a/
        #     shared/
        #   dir_b/
        #     shared/  <- same name
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        dir_a = tmp_path / "dir_a"
        dir_a.mkdir()
        (dir_a / "shared").mkdir()

        dir_b = tmp_path / "dir_b"
        dir_b.mkdir()
        (dir_b / "shared").mkdir()

        workspace = Workspace(external_dirs=["../dir_a/shared", "../dir_b/shared"])

        with pytest.raises(ValueError, match="Name collision"):
            deployment._gather_training_dir(config_path.parent, workspace=workspace)

    def test_external_dir_does_not_exist_fails(self, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        workspace = Workspace(external_dirs=["../nonexistent"])

        with pytest.raises(ValueError, match="does not exist"):
            deployment._gather_training_dir(config_path.parent, workspace=workspace)

    def test_external_dir_is_file_not_directory_fails(self, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        # Create a file instead of directory
        some_file = tmp_path / "some_file.txt"
        some_file.write_text("not a directory")

        workspace = Workspace(external_dirs=["../some_file.txt"])

        with pytest.raises(ValueError, match="not a directory"):
            deployment._gather_training_dir(config_path.parent, workspace=workspace)


class TestArchiveSizeValidation:
    """Tests for archive size validation."""

    @mock.patch("truss_train.deployment.MAX_ARCHIVE_SIZE_BYTES", 100)  # 100 bytes limit
    @mock.patch("truss.remote.baseten.api.BasetenApi.get_blob_credentials")
    def test_archive_exceeds_max_size_fails(self, get_blob_credentials_mock, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_path = project_dir / "config.py"
        config_path.touch()

        # Create a file larger than 100 bytes
        large_file = project_dir / "large_file.bin"
        large_file.write_bytes(b"x" * 500)

        mock_api = mock.Mock()

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            deployment.prepare_push(
                mock_api,
                config_path.parent,
                TrainingJob(image=Image(base_image="hello-world")),
            )
