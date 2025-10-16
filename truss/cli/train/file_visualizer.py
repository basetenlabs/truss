"""Shared file visualization utilities for training commands."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.tree import Tree

from truss.cli.train import common
from truss.cli.utils.output import console


@dataclass
class FileInfo:
    """Information about a file to display."""

    path: str
    size_bytes: int
    modified: Optional[str] = None
    file_type: Optional[str] = None
    permissions: Optional[str] = None


@dataclass
class VisualizationMetadata:
    """Metadata to display in the header box."""

    fields: Dict[str, str]  # field_name: field_value


@dataclass
class VisualizationConfig:
    """Configuration for file tree visualization."""

    title: str  # Main title (e.g., "Cache for Project: my-project")
    metadata: VisualizationMetadata
    files: List[FileInfo]


class FileTreeVisualizer:
    """Visualizes files in a tree structure with metadata and summary boxes."""

    def __init__(self, config: VisualizationConfig):
        """Initialize the visualizer with configuration.

        Args:
            config: Visualization configuration containing title, metadata, and files.
        """
        self.config = config

    def _create_metadata_panel(self) -> Panel:
        """Create a metadata panel box for the header."""
        lines = []
        for field_name, field_value in self.config.metadata.fields.items():
            lines.append(f"{field_name}: {field_value}")

        content = "\n".join(lines)
        if not content:
            content = "No metadata"

        return Panel(content, title="📋 Metadata", border_style="blue", expand=True)

    def _create_summary_panel(self, total_files: int, total_size: int) -> Panel:
        """Create a summary panel box for the footer."""
        size_str = common.format_bytes_to_human_readable(total_size)
        content = f"📊 Total files: {total_files}\n💾 Total size: {size_str}"

        return Panel(content, title="Summary", border_style="blue", expand=True)

    def _build_tree_structure(self) -> Dict[str, Any]:
        """Build a nested dictionary representing the file tree structure.

        Returns:
            A nested dictionary where directories contain subdirectories/files.
        """
        tree_dict: Dict[str, Any] = {}

        for file_info in self.config.files:
            path_parts = file_info.path.split("/")
            current = tree_dict

            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:
                    # This is the file/final node
                    current[part] = file_info
                else:
                    if part not in current:
                        current[part] = {}
                    elif isinstance(current[part], FileInfo):
                        file_node = current[part]
                        current[part] = {"__file__": file_node}
                    current = current[part]

        return tree_dict

    def _add_tree_nodes(
        self, tree: Tree, tree_dict: Dict[str, Any], parent_path: str = ""
    ) -> None:
        """Recursively add nodes to the Rich tree.

        Args:
            tree: The Rich Tree object to add nodes to.
            tree_dict: The nested dictionary representing the tree structure.
            parent_path: The path of the parent directory (for tracking).
        """
        for name, content in tree_dict.items():
            if name == "__file__" and isinstance(content, FileInfo):
                size_str = common.format_bytes_to_human_readable(content.size_bytes)
                label = f"📄 (file) [green]({size_str})[/green]"
                tree.add(label)
            elif isinstance(content, dict):
                # This is a directory
                dir_branch = tree.add(f"📁 [cyan]{name}/[/cyan]")
                self._add_tree_nodes(dir_branch, content, f"{parent_path}/{name}")
            elif isinstance(content, FileInfo):
                # This is a file
                size_str = common.format_bytes_to_human_readable(content.size_bytes)
                label = f"📄 {name} [green]({size_str})[/green]"
                tree.add(label)

    def display(self) -> None:
        """Display the file tree visualization with metadata and summary."""
        # Display metadata panel
        metadata_panel = self._create_metadata_panel()
        console.print(metadata_panel)
        console.print()

        # Display main title and tree
        console.print(f"📁 {self.config.title}")

        tree = Tree("")
        tree_dict = self._build_tree_structure()
        self._add_tree_nodes(tree, tree_dict)
        console.print(tree)

        # Calculate and display summary
        total_files = len(self.config.files)
        total_size = sum(f.size_bytes for f in self.config.files)

        summary_panel = self._create_summary_panel(total_files, total_size)
        console.print()
        console.print(summary_panel)
