import json
from unittest.mock import patch

from click.testing import CliRunner

from truss.cli.train.slurm import (
    build_login_runtime_config,
    build_sbatch_runtime_config,
    detect_default_project,
    parse_gres,
    push_node,
)


class TestParseGres:
    def test_gpu_colon_format(self):
        assert parse_gres("gpu:8") == 8

    def test_gpu_colon_format_4(self):
        assert parse_gres("gpu:4") == 4

    def test_plain_number(self):
        assert parse_gres("8") == 8

    def test_empty_string(self):
        assert parse_gres("") == 8

    def test_malformed_returns_default(self):
        assert parse_gres("gpu:foo:bar") == 8


class TestBuildLoginRuntimeConfig:
    def test_basic(self):
        config = build_login_runtime_config(
            project="my-project",
            gpus_per_node=4,
            partition="H100",
            self_test=True,
        )
        assert config == {
            "project_name": "my-project",
            "job_name": "slurm-login",
            "gpus_per_node": 4,
            "partition": "H100",
            "self_test": True,
        }


class TestBuildSbatchRuntimeConfig:
    def test_basic(self):
        config = build_sbatch_runtime_config(
            project="my-project",
            job_name="worker-1",
            node_count=3,
            gpus_per_node=8,
            partition="H200",
            sbatch_script="#!/bin/bash\necho hello\n",
        )
        assert config == {
            "project_name": "my-project",
            "job_name": "worker-1",
            "node_count": 3,
            "gpus_per_node": 8,
            "partition": "H200",
            "sbatch_script": "#!/bin/bash\necho hello\n",
        }


class TestDetectDefaultProject:
    def test_reads_from_runtime_config(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text(json.dumps({"project_name": "my-slurm"}))
        with patch("truss.cli.train.slurm.WORKSPACE_RUNTIME_CONFIG", config_file):
            assert detect_default_project() == "my-slurm"

    def test_falls_back_without_file(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        with patch("truss.cli.train.slurm.WORKSPACE_RUNTIME_CONFIG", missing):
            assert detect_default_project() == "slurm-harness"

    def test_falls_back_on_bad_json(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text("not valid json")
        with patch("truss.cli.train.slurm.WORKSPACE_RUNTIME_CONFIG", config_file):
            assert detect_default_project() == "slurm-harness"


class TestPushNode:
    @patch("truss_train.public_api.push")
    def test_push_node_sets_up_temp_dir(self, mock_push):
        mock_push.return_value = {"id": "job-123"}
        runtime_config = {"project_name": "test", "node_count": 1}

        result = push_node("login_node", runtime_config, remote="test-remote")

        assert result == {"id": "job-123"}
        mock_push.assert_called_once()
        call_kwargs = mock_push.call_args
        assert call_kwargs.kwargs["remote"] == "test-remote"
        config_path = call_kwargs.kwargs["config"]
        assert str(config_path).endswith("config.py")


class TestSlurmLoginCLI:
    @patch("truss.cli.train.slurm.push_node")
    @patch("truss.cli.train.slurm.build_login_runtime_config")
    @patch("truss.cli.remote_cli.inquire_remote_name", return_value="baseten")
    def test_login_basic(self, _mock_inquire, mock_build, mock_push):
        mock_build.return_value = {"project_name": "slurm-harness"}
        mock_push.return_value = {"id": "job-abc"}

        from truss.cli.cli import truss_cli

        runner = CliRunner()
        result = runner.invoke(
            truss_cli,
            ["train", "slurm", "login", "--project", "my-proj"],
        )
        assert result.exit_code == 0, result.output
        assert "job-abc" in result.output

    @patch("truss.cli.train.slurm.push_node")
    @patch("truss.cli.train.slurm.build_login_runtime_config")
    @patch("truss.cli.remote_cli.inquire_remote_name", return_value="baseten")
    def test_login_with_self_test(self, _mock_inquire, mock_build, mock_push):
        mock_build.return_value = {}
        mock_push.return_value = {"id": "job-xyz"}

        from truss.cli.cli import truss_cli

        runner = CliRunner()
        result = runner.invoke(truss_cli, ["train", "slurm", "login", "--self-test"])
        assert result.exit_code == 0, result.output
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["self_test"] is True


class TestSlurmSbatchCLI:
    @patch("truss.cli.train.slurm.push_node")
    @patch("truss.cli.train.slurm.build_sbatch_runtime_config")
    @patch("truss.cli.train.slurm.detect_default_project", return_value="slurm-harness")
    @patch("truss.cli.train.slurm.parse_gres", return_value=8)
    @patch("truss.cli.remote_cli.inquire_remote_name", return_value="baseten")
    def test_sbatch_with_wrap(
        self, _mock_inquire, _mock_gres, _mock_detect, mock_build, mock_push
    ):
        mock_build.return_value = {}
        mock_push.return_value = {"id": "job-worker-1"}

        from truss.cli.cli import truss_cli

        runner = CliRunner()
        result = runner.invoke(
            truss_cli, ["train", "slurm", "sbatch", "--wrap", "echo hello"]
        )
        assert result.exit_code == 0, result.output
        assert "job-worker-1" in result.output

    @patch("truss.cli.train.slurm.push_node")
    @patch("truss.cli.train.slurm.build_sbatch_runtime_config")
    @patch("truss.cli.train.slurm.detect_default_project", return_value="slurm-harness")
    @patch("truss.cli.train.slurm.parse_gres", return_value=8)
    @patch("truss.cli.remote_cli.inquire_remote_name", return_value="baseten")
    def test_sbatch_with_script(
        self, _mock_inquire, _mock_gres, _mock_detect, mock_build, mock_push, tmp_path
    ):
        mock_build.return_value = {}
        mock_push.return_value = {"id": "job-worker-2"}

        script_file = tmp_path / "train.sh"
        script_file.write_text("#!/bin/bash\necho train\n")

        from truss.cli.cli import truss_cli

        runner = CliRunner()
        result = runner.invoke(
            truss_cli, ["train", "slurm", "sbatch", str(script_file)]
        )
        assert result.exit_code == 0, result.output
        assert "job-worker-2" in result.output

    @patch("truss.cli.remote_cli.inquire_remote_name", return_value="baseten")
    def test_sbatch_no_script_or_wrap_errors(self, _mock_inquire):
        from truss.cli.cli import truss_cli

        runner = CliRunner()
        result = runner.invoke(truss_cli, ["train", "slurm", "sbatch"])
        assert result.exit_code != 0
