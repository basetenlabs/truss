import logging
import tempfile
from pathlib import Path

from truss.remote.baseten.utils.tar import (
    _human_readable_size,
    create_tar_with_progress_bar,
)

_TAR_LOGGER = "truss.remote.baseten.utils.tar"


def _write(path: Path, num_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * num_bytes)


def test_human_readable_size():
    assert _human_readable_size(5) == "5 B"
    assert _human_readable_size(2048) == "2.0 KB"
    assert _human_readable_size(5 * 1024 * 1024) == "5.0 MB"


def test_debug_logs_bundled_files_largest_first(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "config.yaml", 40)
        _write(root / "model" / "model.py", 30)
        _write(root / "model" / "weights.bin", 5 * 1024 * 1024)

        with caplog.at_level(logging.DEBUG, logger=_TAR_LOGGER):
            create_tar_with_progress_bar(root)

    text = caplog.text
    assert "Packing 3 files" in text
    assert "model/weights.bin (5.0 MB)" in text
    assert "config.yaml (40 B)" in text
    # Largest file is listed before the smaller ones.
    assert text.index("weights.bin") < text.index("config.yaml")


def test_no_file_listing_below_debug(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "config.yaml", 40)

        with caplog.at_level(logging.INFO, logger=_TAR_LOGGER):
            create_tar_with_progress_bar(root)

    assert "Packing" not in caplog.text
