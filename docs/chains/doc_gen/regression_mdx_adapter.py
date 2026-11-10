import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_reference import (  # type: ignore[attr-defined]
    _apply_patch,
    _resolve_raw_out,
    generate_sphinx_docs,
)
from mdx_adapter import (  # type: ignore[attr-defined]
    _raw_text_replacements,
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

    def test_empty_overload_heading_is_dropped_in_any_order(self):
        content = """
### `truss_chains.depends`

Sets a dependency.

### `truss_chains.depends`
"""
        result = extract_and_format_parameters_section(content)
        self.assertEqual(result.count("### `truss_chains.depends`"), 1)
        self.assertIn("Sets a dependency.", result)

    def test_context_manager_type_argument_is_normalized(self):
        content = "* **Return type:**\n  *ContextManager*[None, bool | None]\n"
        result = _raw_text_replacements(content)
        self.assertIn("*ContextManager*[None]", result)
        self.assertNotIn("bool | None", result)

    def test_raw_out_stages_missing_parent_and_rejects_committed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "nested" / "chains-reference.mdx"
            self.assertEqual(_resolve_raw_out(scratch), scratch.resolve())
            self.assertTrue(scratch.parent.is_dir())
        for name in ["generated-reference.mdx", "API-reference.mdx"]:
            with self.assertRaises(ValueError):
                _resolve_raw_out(Path(__file__).with_name(name))

    def test_check_and_raw_out_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "chains-reference.mdx"
            with self.assertRaises(ValueError):
                generate_sphinx_docs(Path(directory), check=True, raw_out=scratch)

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
