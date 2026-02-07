"""Resolve HuggingFace revisions to commit SHAs."""

import logging
import re
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


def resolve_hf_revision(
    repo_id: str, revision: Optional[str] = None, token: Optional[str] = None
) -> str:
    """Resolve HF revision to commit SHA (mirrors Rust truss-transfer logic).

    This does the same thing as the Rust code in truss-transfer/src/create/hf_metadata.rs:
    - Calls api_repo.info().await
    - Extracts repo_info.sha

    This is used for best-effort revision pinning during push. Callers should handle
    exceptions gracefully and allow the push to proceed even if resolution fails.

    Args:
        repo_id: HuggingFace repo ID (e.g., "meta-llama/Llama-2-7b")
        revision: Branch, tag, or SHA (None = default branch)
        token: Optional HF token for private repos

    Returns:
        Resolved commit SHA (40-char hex string)

    Raises:
        HfHubHTTPError: If repo doesn't exist, revision is invalid, or network issues
    """
    try:
        api = HfApi(token=token)
        repo_info = api.repo_info(repo_id=repo_id, revision=revision, repo_type="model")
        return repo_info.sha
    except HfHubHTTPError as e:
        logger.debug(f"Failed to resolve HF revision for {repo_id}@{revision}: {e}")
        raise


def is_commit_sha(revision: Optional[str]) -> bool:
    """Check if revision is already a 40-character commit SHA."""
    if not revision:
        return False
    return bool(re.match(r"^[0-9a-f]{40}$", revision))


def parse_hf_source(source: str) -> tuple[str, Optional[str]]:
    """Parse HuggingFace source URI into repo_id and revision.

    Args:
        source: URI like "hf://owner/repo@revision" or "hf://owner/repo"

    Returns:
        Tuple of (repo_id, revision or None)

    Examples:
        >>> parse_hf_source("hf://meta-llama/Llama-2-7b@main")
        ("meta-llama/Llama-2-7b", "main")
        >>> parse_hf_source("hf://meta-llama/Llama-2-7b")
        ("meta-llama/Llama-2-7b", None)
    """
    if not source.startswith("hf://"):
        raise ValueError(f"Not a HuggingFace source: {source}")

    # Remove "hf://" prefix
    path = source[5:]

    # Split on @ to get repo_id and revision
    if "@" in path:
        repo_id, revision = path.rsplit("@", 1)
        return repo_id, revision

    return path, None


def build_hf_source(repo_id: str, revision: str) -> str:
    """Build HuggingFace source URI from repo_id and revision.

    Args:
        repo_id: HuggingFace repo ID
        revision: Commit SHA or branch/tag name

    Returns:
        URI like "hf://owner/repo@revision"
    """
    return f"hf://{repo_id}@{revision}"
