from unittest.mock import Mock

import pytest

from truss.contexts.image_builder import cache_warmer


def test_download_file_using_b10cp_raises_when_download_fails(monkeypatch, tmp_path):
    process = Mock()
    process.wait.return_value = 1
    monkeypatch.setattr(
        cache_warmer, "_download_from_url_using_b10cp", lambda *args: process
    )

    with pytest.raises(RuntimeError, match="weights.bin.*exited with code 1"):
        cache_warmer.download_file_using_b10cp(
            "https://example.com/weights.bin", tmp_path / "weights.bin", "weights.bin"
        )
