from pathlib import Path

import pytest

from truss.util.requirements import (
    _is_valid_requirement,
    parse_requirement_string,
    parse_requirements_from_pyproject,
)


@pytest.fixture
def write_pyproject(tmp_path):
    """Helper to write a pyproject.toml with given content and return its path."""

    def _write(content: str) -> Path:
        path = tmp_path / "pyproject.toml"
        path.write_text(content)
        return path

    return _write


def test_parse_basic_dependencies(write_pyproject):
    path = write_pyproject(
        """
[project]
dependencies = [
    "requests>=2.28",
    "numpy==1.24.0",
    "torch",
]
"""
    )
    result = parse_requirements_from_pyproject(path)
    assert result == ["requests>=2.28", "numpy==1.24.0", "torch"]


def test_parse_empty_dependencies(write_pyproject):
    path = write_pyproject(
        """
[project]
dependencies = []
"""
    )
    assert parse_requirements_from_pyproject(path) == []


def test_parse_missing_dependencies_key(write_pyproject):
    path = write_pyproject(
        """
[project]
name = "my-package"
"""
    )
    assert parse_requirements_from_pyproject(path) == []


def test_parse_missing_project_section(write_pyproject):
    path = write_pyproject(
        """
[tool.ruff]
line-length = 88
"""
    )
    assert parse_requirements_from_pyproject(path) == []


def test_parse_includes_url_references(write_pyproject):
    path = write_pyproject(
        """
[project]
dependencies = [
    "requests>=2.28",
    "my-package @ https://example.com/my-package.tar.gz",
    "numpy==1.24.0",
]
"""
    )
    result = parse_requirements_from_pyproject(path)
    assert result == [
        "requests>=2.28",
        "my-package @ https://example.com/my-package.tar.gz",
        "numpy==1.24.0",
    ]


def test_parse_filters_local_path_dependencies(write_pyproject):
    path = write_pyproject(
        """
[project]
dependencies = [
    "requests>=2.28",
    "./local_package",
    "/absolute/path/to/package",
    "numpy==1.24.0",
]
"""
    )
    result = parse_requirements_from_pyproject(path)
    assert result == ["requests>=2.28", "numpy==1.24.0"]


def test_parse_dependencies_with_extras(write_pyproject):
    path = write_pyproject(
        """
[project]
dependencies = [
    "requests[security]>=2.28",
    "pandas[sql,excel]>=2.0",
]
"""
    )
    result = parse_requirements_from_pyproject(path)
    assert result == ["requests[security]>=2.28", "pandas[sql,excel]>=2.0"]


def test_parse_ignores_optional_dependencies(write_pyproject):
    """Only [project.dependencies] is parsed, not optional-dependencies."""
    path = write_pyproject(
        """
[project]
dependencies = [
    "requests>=2.28",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]
"""
    )
    result = parse_requirements_from_pyproject(path)
    assert result == ["requests>=2.28"]


def test_parse_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_requirements_from_pyproject(tmp_path / "nonexistent.toml")


@pytest.mark.parametrize(
    "req",
    [
        "requests>=2.28",
        "numpy==1.24.0",
        "torch",
        "pandas[sql]>=2.0",
        "my-package~=1.0",
        "my-package @ https://example.com/pkg.tar.gz",
        "pkg @ file:///local/path",
    ],
)
def test_is_valid_requirement(req):
    assert _is_valid_requirement(req) is True


@pytest.mark.parametrize(
    "req", ["./local_package", "/absolute/path", "../relative/path"]
)
def test_is_not_valid_requirement(req):
    assert _is_valid_requirement(req) is False


def test_parse_requirement_string_valid():
    assert parse_requirement_string("requests>=2.28") == "requests>=2.28"
    assert parse_requirement_string("  torch  ") == "torch"
    assert (
        parse_requirement_string("git+https://github.com/foo/bar.git")
        == "git+https://github.com/foo/bar.git"
    )
    assert parse_requirement_string("./local_path") == "./local_path"


def test_parse_requirement_string_filtered():
    assert parse_requirement_string("") is None
    assert parse_requirement_string("# comment") is None
    assert parse_requirement_string("   ") is None
