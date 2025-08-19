import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from truss.cli.logs.utils import ParsedLog
from truss.cli.train_commands import _get_log_format_type, _save_logs_to_file


@dataclass
class TestGetLogFormatType:
    desc: str
    download: bool
    txt: bool
    json: bool
    expected: str | None


class TestTrainCommands(unittest.TestCase):
    def test_get_log_format_type(self):
        """Test the log format type determination logic."""
        test_cases = [
            TestGetLogFormatType(
                desc="default behavior - no flags set",
                download=False,
                txt=False,
                json=False,
                expected=None,
            ),
            TestGetLogFormatType(
                desc="explicit txt flag",
                download=False,
                txt=True,
                json=False,
                expected="txt",
            ),
            TestGetLogFormatType(
                desc="explicit json flag",
                download=False,
                txt=False,
                json=True,
                expected="json",
            ),
            TestGetLogFormatType(
                desc="download with txt - should default to txt",
                download=True,
                txt=True,
                json=False,
                expected="txt",
            ),
            TestGetLogFormatType(
                desc="download with json - should prioritize json",
                download=True,
                txt=False,
                json=True,
                expected="json",
            ),
            TestGetLogFormatType(
                desc="download alone - should default to txt",
                download=True,
                txt=False,
                json=False,
                expected="txt",
            ),
        ]

        for test_case in test_cases:
            with self.subTest(test_case.desc):
                result = _get_log_format_type(
                    download=test_case.download, txt=test_case.txt, json=test_case.json
                )
                assert result == test_case.expected, f"Failed for {test_case.desc}"

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
                logs,
                "aghilan-anime-generator-project",
                "generate-dogs-job",
                "txt",
                tail=True,
                output_dir=temp_dir,
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
