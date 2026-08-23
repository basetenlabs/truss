from io import StringIO

from rich.console import Console

from truss.cli.cannery.progress import ProgressRenderer


def test_non_terminal_progress_keeps_line_oriented_logs():
    output = StringIO()
    renderer = ProgressRenderer(
        Console(file=output, force_terminal=False, color_system=None, width=120)
    )

    renderer("Cannery pull (download): 0/2 files")
    renderer("Cannery pull (download): 2/2 files")
    renderer.close()

    assert output.getvalue().splitlines() == [
        "Cannery pull (download): 0/2 files",
        "Cannery pull (download): 2/2 files",
    ]


def test_terminal_progress_updates_one_live_line():
    output = StringIO()
    renderer = ProgressRenderer(
        Console(file=output, force_terminal=True, color_system=None, width=120)
    )

    renderer("Cannery pull (download): 0/2 files")
    renderer("Cannery pull (download): 1/2 files")
    renderer("Cannery pull (download): 2/2 files")
    renderer.close()

    rendered = output.getvalue()
    assert "Cannery pull (download): 0/2 files" not in rendered
    assert "Cannery pull (download): 1/2 files" not in rendered
    assert "Cannery pull (download): 2/2 files" in rendered
    assert rendered.count("\n") == 1
