import hashlib
import os
from pathlib import Path
from unittest.mock import Mock

import click
import pytest

from truss.cli.cannery import binary


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        url: str = "https://baseten-public.s3.amazonaws.com/cannery",
        history=None,
    ):
        self._content = content
        self.url = url
        self.history = history or []

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        midpoint = len(self._content) // 2
        yield self._content[:midpoint]
        yield self._content[midpoint:]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _artifact(content: bytes = b"cannery-binary"):
    return binary.ArtifactMetadata(
        cannery_version="0.1.0",
        protocol_version=1,
        operating_system="macos",
        architecture="arm64",
        url="https://baseten-public.s3.amazonaws.com/cannery",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_current_platform_normalizes_exact_metadata(monkeypatch):
    monkeypatch.setattr(binary.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(binary.platform, "machine", lambda: "aarch64")

    assert binary.current_platform() == ("macos", "arm64")


def test_download_is_verified_private_atomic_and_proxy_compatible(
    monkeypatch, tmp_path
):
    content = b"cannery-binary"
    artifact = _artifact(content)
    response = FakeResponse(content)
    http_client = Mock()
    http_client.get.return_value = response
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8443")
    monkeypatch.setenv("NO_PROXY", "localhost")

    resolved = binary.resolve_artifact(
        artifact, cache_dir=tmp_path / "cache", http_client=http_client
    )

    assert resolved.name == f"cannery-{artifact.sha256}"
    assert resolved.read_bytes() == content
    if os.name != "nt":
        assert resolved.stat().st_mode & 0o777 == 0o700
        assert resolved.parent.stat().st_mode & 0o777 == 0o700
    assert not list(resolved.parent.glob(".cannery-download-*"))
    request_kwargs = http_client.get.call_args.kwargs
    assert request_kwargs["stream"] is True
    assert request_kwargs["allow_redirects"] is True
    assert "proxies" not in request_kwargs
    assert "verify" not in request_kwargs


def test_resolution_selects_exact_pinned_platform(monkeypatch, tmp_path):
    content = b"cannery-binary"
    artifact = _artifact(content)
    monkeypatch.setattr(binary, "current_platform", lambda: ("macos", "arm64"))

    resolved = binary.resolve_cannery_binary(
        artifacts={artifact.platform_key: artifact},
        cache_dir=tmp_path / "cache",
        http_client=Mock(get=Mock(return_value=FakeResponse(content))),
    )

    assert Path(resolved).name == f"cannery-{artifact.sha256}"


def test_valid_cache_entry_is_reused_without_network(tmp_path):
    content = b"cannery-binary"
    artifact = _artifact(content)
    http_client = Mock()
    first = binary.resolve_artifact(
        artifact,
        cache_dir=tmp_path / "cache",
        http_client=Mock(get=Mock(return_value=FakeResponse(content))),
    )

    second = binary.resolve_artifact(
        artifact, cache_dir=tmp_path / "cache", http_client=http_client
    )

    assert second == first
    http_client.get.assert_not_called()


def test_development_binary_diagnostics_hash_executable(tmp_path):
    executable = tmp_path / "cannery"
    executable.write_bytes(b"development-binary")

    version, digest = binary.binary_diagnostic_metadata(executable)

    assert version == "development-override"
    assert digest == hashlib.sha256(b"development-binary").hexdigest()


def test_digest_mismatch_is_rejected_and_temp_removed(tmp_path):
    artifact = _artifact(b"expected")
    cache = tmp_path / "cache"

    with pytest.raises(click.ClickException, match="SHA-256"):
        binary.resolve_artifact(
            artifact,
            cache_dir=cache,
            http_client=Mock(get=Mock(return_value=FakeResponse(b"modified"))),
        )

    assert not list(cache.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink support is deferred")
def test_cached_symlink_is_rejected(tmp_path):
    artifact = _artifact()
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_bytes(b"cannery-binary")
    cached = cache / f"cannery-{artifact.sha256}"
    cached.symlink_to(target)

    with pytest.raises(click.ClickException, match="symlink"):
        binary.resolve_artifact(artifact, cache_dir=cache)


@pytest.mark.skipif(os.name == "nt", reason="Unix permission bits")
def test_cached_group_writable_file_is_rejected(tmp_path):
    artifact = _artifact()
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cached = cache / f"cannery-{artifact.sha256}"
    cached.write_bytes(b"cannery-binary")
    cached.chmod(0o720)

    with pytest.raises(click.ClickException, match="group- or world-writable"):
        binary.resolve_artifact(artifact, cache_dir=cache)


@pytest.mark.skipif(os.name == "nt", reason="Unix permission bits")
def test_non_private_cache_directory_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o755)

    with pytest.raises(click.ClickException, match="owner-only"):
        binary.resolve_artifact(_artifact(), cache_dir=cache)


def test_remote_resolution_does_not_use_path_fallback(monkeypatch):
    monkeypatch.setattr(binary, "current_platform", lambda: ("macos", "arm64"))
    which = Mock(return_value="/usr/local/bin/cannery")
    monkeypatch.setattr(binary.shutil, "which", which)

    with pytest.raises(click.ClickException, match="no pinned"):
        binary.resolve_cannery_binary(allow_path_fallback=False, artifacts={})

    which.assert_not_called()


@pytest.mark.parametrize(
    "history",
    [
        [Mock(url="https://redirect.example/one")],
        [
            Mock(url="https://redirect.example/one"),
            Mock(url="https://redirect.example/two"),
        ],
    ],
)
def test_download_rejects_direct_and_multi_hop_https_downgrade(tmp_path, history):
    content = b"cannery-binary"
    artifact = _artifact(content)
    response = FakeResponse(
        content, url="http://downloads.example/cannery", history=history
    )

    with pytest.raises(click.ClickException, match="non-HTTPS"):
        binary.resolve_artifact(
            artifact,
            cache_dir=tmp_path / "cache",
            http_client=Mock(get=Mock(return_value=response)),
        )

    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.skipif(os.name == "nt", reason="Unix ownership check")
def test_cached_unowned_file_is_rejected(monkeypatch, tmp_path):
    artifact = _artifact()
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cached = cache / f"cannery-{artifact.sha256}"
    cached.write_bytes(b"cannery-binary")
    cached.chmod(0o700)
    real_lstat = Path.lstat

    def fake_lstat(path):
        result = real_lstat(path)
        if path == cached:
            values = list(result)
            values[4] = os.getuid() + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(click.ClickException, match="current user"):
        binary.resolve_artifact(artifact, cache_dir=cache)
