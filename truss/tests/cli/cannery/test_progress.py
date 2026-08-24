from io import StringIO

from rich.console import Console

from truss.cli.cannery.progress import ProgressRenderer


def test_non_terminal_subset_progress_keeps_selected_totals_line_oriented():
    output = StringIO()
    renderer = ProgressRenderer(
        Console(file=output, force_terminal=False, color_system=None, width=120)
    )

    renderer("Cannery pull (download): 1/3 files, 67108864/268435456 bytes")
    renderer("Cannery pull (download): 3/3 files, 268435456/268435456 bytes")
    renderer.close()

    assert output.getvalue().splitlines() == [
        "Cannery pull (download): 1/3 files, 67108864/268435456 bytes",
        "Cannery pull (download): 3/3 files, 268435456/268435456 bytes",
    ]


def test_terminal_subset_progress_updates_one_live_line():
    output = StringIO()
    renderer = ProgressRenderer(
        Console(file=output, force_terminal=True, color_system=None, width=120)
    )

    renderer("Cannery pull (download): 1/3 files, 67108864/268435456 bytes")
    renderer("Cannery pull (download): 2/3 files, 134217728/268435456 bytes")
    renderer("Cannery pull (download): 3/3 files, 268435456/268435456 bytes")
    renderer.close()

    rendered = output.getvalue()
    assert "1/3 files, 67108864/268435456 bytes" not in rendered
    assert "2/3 files, 134217728/268435456 bytes" not in rendered
    assert "3/3 files, 268435456/268435456 bytes" in rendered
    assert rendered.count("\n") == 1
