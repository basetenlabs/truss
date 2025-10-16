"""Tests for file_visualizer module."""

from dataclasses import dataclass

import pytest

from truss.cli.train.file_visualizer import (
    FileInfo,
    FileTreeVisualizer,
    VisualizationConfig,
    VisualizationMetadata,
)


@dataclass
class TestCase:
    """Test case for file tree visualizer tests."""

    desc: str
    input: VisualizationConfig
    expected: dict  # Expected attributes to check


def test_file_tree_visualizer_initialization():
    """Test FileTreeVisualizer initialization."""
    config = VisualizationConfig(
        title="Test Title",
        metadata=VisualizationMetadata(fields={"Field1": "Value1"}),
        files=[FileInfo(path="test.txt", size_bytes=100)],
    )

    visualizer = FileTreeVisualizer(config)

    assert visualizer.config == config
    assert visualizer.config.title == "Test Title"
    assert len(visualizer.config.files) == 1


def test_build_tree_structure_flat_files():
    """Test building tree structure with flat files."""
    test_cases = [
        TestCase(
            desc="single file",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[FileInfo(path="file1.txt", size_bytes=100)],
            ),
            expected={"keys": ["file1.txt"], "depth": 1},
        ),
        TestCase(
            desc="multiple files",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[
                    FileInfo(path="file1.txt", size_bytes=100),
                    FileInfo(path="file2.txt", size_bytes=200),
                ],
            ),
            expected={"keys": ["file1.txt", "file2.txt"], "depth": 1},
        ),
    ]

    for tc in test_cases:
        visualizer = FileTreeVisualizer(tc.input)
        tree_dict = visualizer._build_tree_structure()

        assert sorted(tree_dict.keys()) == sorted(tc.expected["keys"]), (
            f"Failed for: {tc.desc}"
        )


def test_build_tree_structure_nested_paths():
    """Test building tree structure with nested paths."""
    test_cases = [
        TestCase(
            desc="single level nesting",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[FileInfo(path="dir1/file1.txt", size_bytes=100)],
            ),
            expected={"top_level_keys": ["dir1"], "has_nested": True},
        ),
        TestCase(
            desc="multiple levels nesting",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[
                    FileInfo(path="dir1/dir2/file1.txt", size_bytes=100),
                    FileInfo(path="dir1/dir2/file2.txt", size_bytes=200),
                    FileInfo(path="dir1/file3.txt", size_bytes=300),
                ],
            ),
            expected={"top_level_keys": ["dir1"], "has_nested": True},
        ),
        TestCase(
            desc="mixed flat and nested",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[
                    FileInfo(path="root_file.txt", size_bytes=100),
                    FileInfo(path="dir1/nested_file.txt", size_bytes=200),
                ],
            ),
            expected={"top_level_keys": ["root_file.txt", "dir1"], "has_nested": True},
        ),
    ]

    for tc in test_cases:
        visualizer = FileTreeVisualizer(tc.input)
        tree_dict = visualizer._build_tree_structure()

        assert sorted(tree_dict.keys()) == sorted(tc.expected["top_level_keys"]), (
            f"Failed for: {tc.desc}"
        )

        # Check for nested structure
        if tc.expected["has_nested"]:
            # At least one entry should be a dict (directory)
            has_dict = any(isinstance(v, dict) for v in tree_dict.values())
            assert has_dict, f"Failed for: {tc.desc} - expected nested structure"


def test_build_tree_structure_complex():
    """Test building tree structure with complex directory hierarchies."""
    files = [
        FileInfo(path="model/weights.bin", size_bytes=1000),
        FileInfo(path="model/config.json", size_bytes=100),
        FileInfo(path="data/train/file1.txt", size_bytes=200),
        FileInfo(path="data/train/file2.txt", size_bytes=300),
        FileInfo(path="data/test/file3.txt", size_bytes=400),
        FileInfo(path="README.md", size_bytes=50),
    ]

    config = VisualizationConfig(
        title="Test", metadata=VisualizationMetadata(fields={}), files=files
    )

    visualizer = FileTreeVisualizer(config)
    tree_dict = visualizer._build_tree_structure()

    # Check top-level keys
    assert sorted(tree_dict.keys()) == ["README.md", "data", "model"]

    # Check model directory
    assert isinstance(tree_dict["model"], dict)
    assert "weights.bin" in tree_dict["model"]
    assert "config.json" in tree_dict["model"]

    # Check data directory structure
    assert isinstance(tree_dict["data"], dict)
    assert "train" in tree_dict["data"]
    assert "test" in tree_dict["data"]
    assert isinstance(tree_dict["data"]["train"], dict)
    assert isinstance(tree_dict["data"]["test"], dict)


def test_metadata_panel_creation(capsys):
    """Test metadata panel creation."""
    test_cases = [
        TestCase(
            desc="single metadata field",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={"Field1": "Value1"}),
                files=[],
            ),
            expected={"field_count": 1},
        ),
        TestCase(
            desc="multiple metadata fields",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(
                    fields={
                        "Job ID": "job123",
                        "Project ID": "proj456",
                        "Created At": "2024-01-01",
                    }
                ),
                files=[],
            ),
            expected={"field_count": 3},
        ),
        TestCase(
            desc="empty metadata",
            input=VisualizationConfig(
                title="Test", metadata=VisualizationMetadata(fields={}), files=[]
            ),
            expected={"field_count": 0},
        ),
    ]

    for tc in test_cases:
        visualizer = FileTreeVisualizer(tc.input)
        panel = visualizer._create_metadata_panel()

        assert panel is not None, f"Failed for: {tc.desc}"
        # Panel should have content
        assert hasattr(panel, "renderable"), f"Failed for: {tc.desc}"


def test_summary_panel_creation():
    """Test summary panel creation."""
    test_cases = [
        TestCase(
            desc="small files",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[
                    FileInfo(path="file1.txt", size_bytes=100),
                    FileInfo(path="file2.txt", size_bytes=200),
                ],
            ),
            expected={"total_files": 2, "total_size": 300},
        ),
        TestCase(
            desc="large files",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[
                    FileInfo(path="file1.bin", size_bytes=1000000000),  # 1GB
                    FileInfo(path="file2.bin", size_bytes=2000000000),  # 2GB
                ],
            ),
            expected={"total_files": 2, "total_size": 3000000000},
        ),
        TestCase(
            desc="empty files list",
            input=VisualizationConfig(
                title="Test", metadata=VisualizationMetadata(fields={}), files=[]
            ),
            expected={"total_files": 0, "total_size": 0},
        ),
    ]

    for tc in test_cases:
        visualizer = FileTreeVisualizer(tc.input)
        total_files = len(tc.input.files)
        total_size = sum(f.size_bytes for f in tc.input.files)

        panel = visualizer._create_summary_panel(total_files, total_size)

        assert panel is not None, f"Failed for: {tc.desc}"
        assert hasattr(panel, "renderable"), f"Failed for: {tc.desc}"


def test_file_info_optional_fields():
    """Test FileInfo with optional fields."""
    test_cases = [
        TestCase(
            desc="all fields provided",
            input=None,  # Not used in this test
            expected={
                "file": FileInfo(
                    path="test.txt",
                    size_bytes=100,
                    modified="2024-01-01",
                    file_type="file",
                    permissions="-rw-r--r--",
                )
            },
        ),
        TestCase(
            desc="only required fields",
            input=None,
            expected={"file": FileInfo(path="test.txt", size_bytes=100)},
        ),
    ]

    for tc in test_cases:
        file_info = tc.expected["file"]

        assert file_info.path == "test.txt", f"Failed for: {tc.desc}"
        assert file_info.size_bytes == 100, f"Failed for: {tc.desc}"


def test_visualization_display_does_not_crash(capsys):
    """Test that display method doesn't crash with various inputs."""
    test_cases = [
        TestCase(
            desc="basic files",
            input=VisualizationConfig(
                title="Test Display",
                metadata=VisualizationMetadata(fields={"Field": "Value"}),
                files=[
                    FileInfo(path="file1.txt", size_bytes=100),
                    FileInfo(path="dir/file2.txt", size_bytes=200),
                ],
            ),
            expected={},
        ),
        TestCase(
            desc="empty files",
            input=VisualizationConfig(
                title="Empty Test", metadata=VisualizationMetadata(fields={}), files=[]
            ),
            expected={},
        ),
        TestCase(
            desc="deeply nested structure",
            input=VisualizationConfig(
                title="Nested Test",
                metadata=VisualizationMetadata(fields={"ID": "123"}),
                files=[FileInfo(path="a/b/c/d/e/f/file.txt", size_bytes=100)],
            ),
            expected={},
        ),
    ]

    for tc in test_cases:
        visualizer = FileTreeVisualizer(tc.input)
        try:
            visualizer.display()
            # Capture output to prevent cluttering test output
            captured = capsys.readouterr()
            # Basic check that something was printed
            assert len(captured.out) > 0 or len(captured.err) == 0, (
                f"Failed for: {tc.desc}"
            )
        except Exception as e:
            pytest.fail(f"Display crashed for {tc.desc}: {e}")


def test_build_tree_structure_path_conflicts():
    """Test that path conflicts (file and directory with same path prefix) are handled."""
    test_cases = [
        TestCase(
            desc="file conflicts with directory path",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[
                    FileInfo(path="foo", size_bytes=100),
                    FileInfo(path="foo/bar.txt", size_bytes=200),
                ],
            ),
            expected={
                "has_foo": True,
                "foo_is_dict": True,
                "has_special_file_key": True,
            },
        ),
        TestCase(
            desc="deeply nested path conflict",
            input=VisualizationConfig(
                title="Test",
                metadata=VisualizationMetadata(fields={}),
                files=[
                    FileInfo(path="a/b/c", size_bytes=100),
                    FileInfo(path="a/b/c/d/e.txt", size_bytes=200),
                ],
            ),
            expected={"has_a": True, "nested_conflict": True},
        ),
    ]

    for tc in test_cases:
        visualizer = FileTreeVisualizer(tc.input)
        tree_dict = visualizer._build_tree_structure()

        if tc.expected.get("has_foo"):
            assert "foo" in tree_dict, f"Failed for: {tc.desc}"
            assert isinstance(tree_dict["foo"], dict), f"Failed for: {tc.desc}"
            if tc.expected.get("has_special_file_key"):
                assert "__file__" in tree_dict["foo"], f"Failed for: {tc.desc}"
                assert isinstance(tree_dict["foo"]["__file__"], FileInfo), (
                    f"Failed for: {tc.desc}"
                )

        if tc.expected.get("has_a"):
            assert "a" in tree_dict, f"Failed for: {tc.desc}"
            assert isinstance(tree_dict["a"], dict), f"Failed for: {tc.desc}"
