import json
import sys

import rich
import rich.live
import rich.logging
import rich.spinner
import rich.table
import rich.traceback
from rich.console import Console

rich.spinner.SPINNERS["deploying"] = {"interval": 500, "frames": ["👾 ", " 👾"]}
rich.spinner.SPINNERS["building"] = {"interval": 500, "frames": ["🛠️ ", " 🛠️"]}
rich.spinner.SPINNERS["loading"] = {"interval": 500, "frames": ["⏱️ ", " ⏱️"]}
rich.spinner.SPINNERS["active"] = {"interval": 500, "frames": ["💚 ", " 💚"]}
rich.spinner.SPINNERS["failed"] = {"interval": 500, "frames": ["😤 ", " 😤"]}


console = Console()
error_console = Console(stderr=True, style="bold red")


def json_output(data: object) -> None:
    """Write structured JSON to stdout for machine consumption."""
    print(json.dumps(data), file=sys.stdout, flush=True)
