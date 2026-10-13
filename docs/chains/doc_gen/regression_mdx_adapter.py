import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_reference import _apply_patch  # type: ignore[attr-defined]
from mdx_adapter import (  # type: ignore[attr-defined]
    extract_and_format_parameters_section,
)


class OverloadedFunctionTests(unittest.TestCase):
    def test_overloaded_function_keeps_released_parameter_types(self):
        content = """
### `truss_chains.depends`

Sets a dependency.

* **Parameters:**
  * **retries** – Retries the request.
"""
        result = extract_and_format_parameters_section(content)
        self.assertIn("*int*", result)

    def test_empty_overload_heading_is_not_repeated(self):
        content = """
### `truss_chains.depends`

### `truss_chains.depends`

Sets a dependency.
"""
        result = extract_and_format_parameters_section(content)
        self.assertEqual(result.count("### `truss_chains.depends`"), 1)

    def test_failed_curation_preserves_output_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.mdx"
            output.write_text("Existing curated reference")
            failure = subprocess.CalledProcessError(1, ["patch"], stderr="stale hunk")
            with (
                patch("generate_reference.subprocess.run", side_effect=failure),
                self.assertRaises(RuntimeError),
            ):
                _apply_patch("generated.mdx", "reference.patch", str(output))
            self.assertEqual(output.read_text(), "Existing curated reference")


if __name__ == "__main__":
    unittest.main()
