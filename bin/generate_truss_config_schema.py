"""Generate JSON Schema for truss config.yaml.

Usage:
    uv run bin/generate_truss_config_schema.py            # write schema
    uv run bin/generate_truss_config_schema.py --check    # check if up to date
"""

import json
import sys
from pathlib import Path

from truss.base.truss_config import TrussConfig

SCHEMA_PATH = Path(__file__).parent.parent / "truss" / "config.schema.json"


def generate() -> str:
    schema = TrussConfig.model_json_schema()
    return json.dumps(schema, indent=2) + "\n"


def main() -> None:
    check = "--check" in sys.argv
    new_schema = generate()

    if check:
        if not SCHEMA_PATH.exists():
            print(
                f"{SCHEMA_PATH} does not exist. Run `uv run bin/generate_truss_config_schema.py` to generate."
            )
            sys.exit(1)
        current = SCHEMA_PATH.read_text()
        if current != new_schema:
            print(
                f"{SCHEMA_PATH} is out of date. Run `uv run bin/generate_truss_config_schema.py` to regenerate."
            )
            sys.exit(1)
        print("Schema is up to date.")
    else:
        SCHEMA_PATH.write_text(new_schema)
        print(f"Wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
