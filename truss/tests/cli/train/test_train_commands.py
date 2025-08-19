import json
import tempfile
from pathlib import Path

from truss.cli.logs.utils import ParsedLog
from truss.cli.train_commands import _get_log_format_type, _save_logs_to_file


class TestTrainCommands:
    def test_get_log_format_type(self):
        """Test the log format type determination logic."""
        # Test default behavior
        assert _get_log_format_type(download=False, txt=False, json=False) is None

        # Test explicit txt
        assert _get_log_format_type(download=False, txt=True, json=False) == "txt"

        # Test explicit json
        assert _get_log_format_type(download=False, txt=False, json=True) == "json"

        # Test download with txt (should default to txt)
        assert _get_log_format_type(download=True, txt=True, json=False) == "txt"

        # Test download with json (should prioritize json)
        assert _get_log_format_type(download=True, txt=False, json=True) == "json"

        # Test download alone (should default to txt)
        assert _get_log_format_type(download=True, txt=False, json=False) == "txt"

    def test_save_logs_to_file_txt(self):
        """Test saving logs in text format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample logs
            logs = [
                ParsedLog(
                    timestamp="1640995200000000000",
                    message="Test log 1",
                    replica="replica-1",
                ),
                ParsedLog(
                    timestamp="1640995260000000000", message="Test log 2", replica=None
                ),
            ]

            filename = _save_logs_to_file(
                logs, "test-project", "test-job", "txt", tail=False, output_dir=temp_dir
            )

            # Check file was created
            file_path = Path(temp_dir) / filename
            assert file_path.exists()

            # Check content
            content = file_path.read_text()
            assert "Test log 1" in content
            assert "Test log 2" in content
            assert "(replica-1)" in content
            assert (
                "2021-12-31" in content
            )  # Check timestamp formatting (correct date for the test timestamp)

    def test_save_logs_to_file_json(self):
        """Test saving logs in JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample logs
            logs = [
                ParsedLog(
                    timestamp="1640995200000000000",
                    message="Test log 1",
                    replica="replica-1",
                ),
                ParsedLog(
                    timestamp="1640995260000000000", message="Test log 2", replica=None
                ),
            ]

            filename = _save_logs_to_file(
                logs,
                "test-project",
                "test-job",
                "json",
                tail=False,
                output_dir=temp_dir,
            )

            # Check file was created
            file_path = Path(temp_dir) / filename
            assert file_path.exists()

            # Check content
            content = json.loads(file_path.read_text())
            assert len(content) == 2
            assert content[0]["message"] == "Test log 1"
            assert content[0]["replica"] == "replica-1"
            assert content[1]["message"] == "Test log 2"
            assert content[1]["replica"] is None

    def test_save_logs_to_file_tail_mode(self):
        """Test saving logs in tail mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample logs
            logs = [
                ParsedLog(
                    timestamp="1640995200000000000",
                    message="Test log 1",
                    replica="replica-1",
                )
            ]

            filename = _save_logs_to_file(
                logs, "test-project", "test-job", "txt", tail=True, output_dir=temp_dir
            )

            # Check filename contains tail suffix
            assert "_tail_" in filename

            # Check file was created
            file_path = Path(temp_dir) / filename
            assert file_path.exists()

    def test_filename_formatting(self):
        """Test that filenames are properly formatted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logs = [
                ParsedLog(
                    timestamp="1640995200000000000", message="Test log", replica=None
                )
            ]

            filename = _save_logs_to_file(
                logs, "my-project", "my-job", "txt", tail=False, output_dir=temp_dir
            )

            # Check filename format
            assert filename.startswith("training_logs_my-job_")
            assert filename.endswith(".txt")

            # Check timestamp format
            timestamp_part = filename.split("_")[-1].replace(".txt", "")
            assert len(timestamp_part) == 6  # HHMMSS format (time part only)
