"""Tests for HuggingFace revision resolver."""

from unittest.mock import Mock, patch

import pytest

from truss.util.hf_revision_resolver import (
    build_hf_source,
    is_commit_sha,
    parse_hf_source,
    resolve_hf_revision,
)


def test_parse_hf_source_with_revision():
    """Test parsing HF source with explicit revision."""
    repo_id, revision = parse_hf_source("hf://meta-llama/Llama-2-7b@main")
    assert repo_id == "meta-llama/Llama-2-7b"
    assert revision == "main"


def test_parse_hf_source_with_sha():
    """Test parsing HF source with commit SHA."""
    sha = "a" * 40
    repo_id, revision = parse_hf_source(f"hf://meta-llama/Llama-2-7b@{sha}")
    assert repo_id == "meta-llama/Llama-2-7b"
    assert revision == sha


def test_parse_hf_source_without_revision():
    """Test parsing HF source without revision (should return None)."""
    repo_id, revision = parse_hf_source("hf://meta-llama/Llama-2-7b")
    assert repo_id == "meta-llama/Llama-2-7b"
    assert revision is None


def test_parse_hf_source_invalid():
    """Test parsing non-HF source raises ValueError."""
    with pytest.raises(ValueError, match="Not a HuggingFace source"):
        parse_hf_source("s3://bucket/path")

    with pytest.raises(ValueError, match="Not a HuggingFace source"):
        parse_hf_source("gs://bucket/path")


def test_is_commit_sha():
    """Test commit SHA detection."""
    # Valid SHAs
    assert is_commit_sha("a" * 40)
    assert is_commit_sha("0123456789abcdef" * 2 + "01234567")
    assert is_commit_sha("f" * 40)

    # Invalid SHAs
    assert not is_commit_sha("main")
    assert not is_commit_sha("v1.0.0")
    assert not is_commit_sha("a" * 39)  # Too short
    assert not is_commit_sha("a" * 41)  # Too long
    assert not is_commit_sha(None)
    assert not is_commit_sha("")
    assert not is_commit_sha("gggggggggggggggggggggggggggggggggggggggg")  # Invalid hex


def test_build_hf_source():
    """Test building HF source URI."""
    sha = "abc1230000000000000000000000000000000000"
    uri = build_hf_source("meta-llama/Llama-2-7b", sha)
    assert uri == f"hf://meta-llama/Llama-2-7b@{sha}"


def test_build_hf_source_with_branch():
    """Test building HF source URI with branch name."""
    uri = build_hf_source("meta-llama/Llama-2-7b", "main")
    assert uri == "hf://meta-llama/Llama-2-7b@main"


@patch("truss.util.hf_revision_resolver.HfApi")
def test_resolve_hf_revision(mock_hf_api):
    """Test resolving HF revision to SHA."""
    mock_repo_info = Mock()
    mock_repo_info.sha = "a" * 40
    mock_hf_api.return_value.repo_info.return_value = mock_repo_info

    sha = resolve_hf_revision("meta-llama/Llama-2-7b", "main")
    assert sha == "a" * 40

    # Verify API was called correctly
    mock_hf_api.assert_called_once_with(token=None)
    mock_hf_api.return_value.repo_info.assert_called_once_with(
        repo_id="meta-llama/Llama-2-7b", revision="main", repo_type="model"
    )


@patch("truss.util.hf_revision_resolver.HfApi")
def test_resolve_hf_revision_without_revision(mock_hf_api):
    """Test resolving HF revision when no revision specified (uses default)."""
    mock_repo_info = Mock()
    mock_repo_info.sha = "b" * 40
    mock_hf_api.return_value.repo_info.return_value = mock_repo_info

    sha = resolve_hf_revision("meta-llama/Llama-2-7b", None)
    assert sha == "b" * 40

    # Verify revision=None was passed
    mock_hf_api.return_value.repo_info.assert_called_once_with(
        repo_id="meta-llama/Llama-2-7b", revision=None, repo_type="model"
    )


@patch("truss.util.hf_revision_resolver.HfApi")
def test_resolve_hf_revision_with_token(mock_hf_api):
    """Test resolving HF revision with auth token."""
    mock_repo_info = Mock()
    mock_repo_info.sha = "c" * 40
    mock_hf_api.return_value.repo_info.return_value = mock_repo_info

    sha = resolve_hf_revision("meta-llama/Llama-2-7b", "main", token="hf_test_token")
    assert sha == "c" * 40

    # Verify token was passed to HfApi
    mock_hf_api.assert_called_once_with(token="hf_test_token")
