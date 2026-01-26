import csv
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import rich

from truss.cli.train import common
from truss.cli.utils import common as cli_common
from truss.cli.utils.output import console
from truss.remote.baseten.custom_types import (
    FileSummary,
    FileSummaryWithTotalSize,
    GetCacheSummaryResponseV1,
)
from truss.remote.baseten.remote import BasetenRemote

# Sort constants
SORT_BY_FILEPATH = "filepath"
SORT_BY_SIZE = "size"
SORT_BY_MODIFIED = "modified"
SORT_BY_TYPE = "type"
SORT_BY_PERMISSIONS = "permissions"
SORT_ORDER_ASC = "asc"
SORT_ORDER_DESC = "desc"

# Output format constants
OUTPUT_FORMAT_CLI_TABLE = "cli-table"
OUTPUT_FORMAT_CSV = "csv"
OUTPUT_FORMAT_JSON = "json"


def calculate_directory_sizes(
    files: list[FileSummary], max_depth: int = 100
) -> dict[str, int]:
    """Calculate total sizes for directories based on their file contents."""
    directory_sizes = {}

    for file_info in files:
        if file_info.file_type == "directory":
            directory_sizes[file_info.path] = 0

    for file_info in files:
        current_path = file_info.path
        for i in range(max_depth):
            if current_path is None:
                break
            if current_path in directory_sizes:
                directory_sizes[current_path] += file_info.size_bytes
            # Move to parent directory
            parent = os.path.dirname(current_path)
            if parent == current_path:  # Reached root
                break
            current_path = parent

    return directory_sizes


def create_file_summary_with_directory_sizes(
    files: list[FileSummary],
) -> list[FileSummaryWithTotalSize]:
    """Create file summaries with total sizes including directory sizes."""
    directory_sizes = calculate_directory_sizes(files)
    return [
        FileSummaryWithTotalSize(
            file_summary=file_info,
            total_size=directory_sizes.get(file_info.path, file_info.size_bytes),
        )
        for file_info in files
    ]


def _get_sort_key(sort_by: str) -> Callable[[FileSummaryWithTotalSize], Any]:
    """Get the sort key function for the given sort option."""
    if sort_by == SORT_BY_FILEPATH:
        return lambda x: x.file_summary.path
    elif sort_by == SORT_BY_SIZE:
        return lambda x: x.total_size
    elif sort_by == SORT_BY_MODIFIED:
        return lambda x: x.file_summary.modified
    elif sort_by == SORT_BY_TYPE:
        return lambda x: x.file_summary.file_type or ""
    elif sort_by == SORT_BY_PERMISSIONS:
        return lambda x: x.file_summary.permissions or ""
    else:
        raise ValueError(f"Invalid --sort argument: {sort_by}")


class CacheSummaryViewer(ABC):
    """Base class for cache summary viewers that output in different formats."""

    @abstractmethod
    def output_cache_summary(
        self,
        cache_data: Optional[GetCacheSummaryResponseV1],
        files_with_total_sizes: list[FileSummaryWithTotalSize],
        total_size: int,
        total_size_str: str,
        project_id: str,
    ) -> None:
        """Output the cache summary in the viewer's format."""
        pass

    @abstractmethod
    def output_no_cache_message(self, project_id: str) -> None:
        """Output message when no cache summary is found."""
        pass


class CLITableViewer(CacheSummaryViewer):
    """Viewer that outputs cache summary as a styled CLI table."""

    def output_cache_summary(
        self,
        cache_data: Optional[GetCacheSummaryResponseV1],
        files_with_total_sizes: list[FileSummaryWithTotalSize],
        total_size: int,
        total_size_str: str,
        project_id: str,
    ) -> None:
        """Output cache summary as a styled CLI table."""
        if not cache_data:
            return

        table = rich.table.Table(title=f"Cache summary for project: {project_id}")
        table.add_column("File Path", style="cyan", overflow="fold")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        table.add_column("Type")
        table.add_column("Permissions", style="magenta")

        console.print(
            f"📅 Cache captured at: {cache_data.timestamp}", style="bold blue"
        )
        console.print(f"📁 Project ID: {cache_data.project_id}", style="bold blue")
        console.print()
        console.print(
            f"📊 Total files: {len(files_with_total_sizes)}", style="bold green"
        )
        console.print(f"💾 Total size: {total_size_str}", style="bold green")
        console.print()
        # Note: Long file paths wrap across multiple lines. To copy full paths easily,
        # use --output-format csv or --output-format json
        if any(len(f.file_summary.path) > 60 for f in files_with_total_sizes):
            console.print(
                "💡 Tip: Use -o csv or -o json to output paths on single lines for easier copying",
                style="dim",
            )
            console.print()

        for file_info in files_with_total_sizes:
            total_size = file_info.total_size

            size_str = cli_common.format_bytes_to_human_readable(int(total_size))

            modified_str = cli_common.format_localized_time(
                file_info.file_summary.modified
            )

            table.add_row(
                file_info.file_summary.path,
                size_str,
                modified_str,
                file_info.file_summary.file_type or "Unknown",
                file_info.file_summary.permissions or "Unknown",
            )

        console.print(table)

    def output_no_cache_message(self, project_id: str) -> None:
        """Output message when no cache summary is found."""
        console.print("No cache summary found for this project.", style="yellow")


class CSVViewer(CacheSummaryViewer):
    """Viewer that outputs cache summary in CSV format."""

    def output_cache_summary(
        self,
        cache_data: Optional[GetCacheSummaryResponseV1],
        files_with_total_sizes: list[FileSummaryWithTotalSize],
        total_size: int,
        total_size_str: str,
        project_id: str,
    ) -> None:
        """Output cache summary in CSV format."""
        writer = csv.writer(sys.stdout)
        # Write header
        writer.writerow(
            [
                "File Path",
                "Size (bytes)",
                "Size (human readable)",
                "Modified",
                "Type",
                "Permissions",
            ]
        )
        # Write data rows
        for file_info in files_with_total_sizes:
            size_str = cli_common.format_bytes_to_human_readable(
                int(file_info.total_size)
            )
            modified_str = cli_common.format_localized_time(
                file_info.file_summary.modified
            )
            writer.writerow(
                [
                    file_info.file_summary.path,
                    str(file_info.total_size),
                    size_str,
                    modified_str,
                    file_info.file_summary.file_type or "Unknown",
                    file_info.file_summary.permissions or "Unknown",
                ]
            )

    def output_no_cache_message(self, project_id: str) -> None:
        """Output empty CSV with headers when no cache summary is found."""
        self.output_cache_summary(None, [], 0, "0 B", project_id)


class JSONViewer(CacheSummaryViewer):
    """Viewer that outputs cache summary in JSON format."""

    def output_cache_summary(
        self,
        cache_data: Optional[GetCacheSummaryResponseV1],
        files_with_total_sizes: list[FileSummaryWithTotalSize],
        total_size: int,
        total_size_str: str,
        project_id: str,
    ) -> None:
        """Output cache summary in JSON format."""
        files_data = []
        for file_info in files_with_total_sizes:
            size_str = cli_common.format_bytes_to_human_readable(
                int(file_info.total_size)
            )
            modified_str = cli_common.format_localized_time(
                file_info.file_summary.modified
            )
            files_data.append(
                {
                    "path": file_info.file_summary.path,
                    "size_bytes": file_info.total_size,
                    "size_human_readable": size_str,
                    "modified": modified_str,
                    "type": file_info.file_summary.file_type or "Unknown",
                    "permissions": file_info.file_summary.permissions or "Unknown",
                }
            )

        output = {
            "timestamp": cache_data.timestamp if cache_data else "",
            "project_id": cache_data.project_id if cache_data else project_id,
            "total_files": len(files_with_total_sizes),
            "total_size_bytes": total_size,
            "total_size_human_readable": total_size_str,
            "files": files_data,
        }

        print(json.dumps(output, indent=2))

    def output_no_cache_message(self, project_id: str) -> None:
        """Output empty JSON structure when no cache summary is found."""
        self.output_cache_summary(None, [], 0, "0 B", project_id)


def _get_cache_summary_viewer(output_format: str) -> CacheSummaryViewer:
    """Factory function to get the appropriate viewer for the output format."""
    if output_format == OUTPUT_FORMAT_CSV:
        return CSVViewer()
    elif output_format == OUTPUT_FORMAT_JSON:
        return JSONViewer()
    else:
        return CLITableViewer()


def view_cache_summary(
    remote_provider: BasetenRemote,
    project_id: str,
    sort_by: str = SORT_BY_FILEPATH,
    order: str = SORT_ORDER_ASC,
    output_format: str = OUTPUT_FORMAT_CLI_TABLE,
):
    """View cache summary for a training project."""
    viewer_factories = {
        OUTPUT_FORMAT_CSV: lambda: CSVViewer(),
        OUTPUT_FORMAT_JSON: lambda: JSONViewer(),
        OUTPUT_FORMAT_CLI_TABLE: lambda: CLITableViewer(),
    }
    viewer_factory = viewer_factories.get(output_format)
    if not viewer_factory:
        raise ValueError(f"Invalid output format: {output_format}")
    viewer = viewer_factory()
    try:
        raw_cache_data = remote_provider.api.get_cache_summary(project_id)

        if not raw_cache_data:
            viewer.output_no_cache_message(project_id)
            return

        cache_data = GetCacheSummaryResponseV1.model_validate(raw_cache_data)

        files = cache_data.file_summaries
        files_with_total_sizes = (
            create_file_summary_with_directory_sizes(files) if files else []
        )

        reverse = order == SORT_ORDER_DESC
        sort_key = _get_sort_key(sort_by)
        files_with_total_sizes.sort(key=sort_key, reverse=reverse)

        total_size = sum(
            file_info.file_summary.size_bytes for file_info in files_with_total_sizes
        )
        total_size_str = common.format_bytes_to_human_readable(total_size)

        viewer.output_cache_summary(
            cache_data, files_with_total_sizes, total_size, total_size_str, project_id
        )

    except Exception as e:
        # For CSV/JSON formats, print to stderr to keep stdout clean for piping
        if output_format in (OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_JSON):
            print(f"Error fetching cache summary: {str(e)}", file=sys.stderr)
        else:
            console.print(f"Error fetching cache summary: {str(e)}", style="red")
        raise
