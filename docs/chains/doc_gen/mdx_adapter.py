# type: ignore  # This tool is only for Marius.
"""Super hacky plugin to make the generated markdown more suitable for
rendering in mintlify as an mdx doc."""

import os
import re
from typing import Any, Dict

from docutils import nodes
from docutils.io import StringOutput
from generate_reference import NON_PUBLIC_SYMBOLS
from sphinx.util.osutil import ensuredir, os_path
from sphinx_markdown_builder import MarkdownBuilder
from sphinx_markdown_builder.translator import MarkdownTranslator

PYDANTIC_DEFAULT_DOCSTRING = """

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.
"""


class MDXAdapterTranslator(MarkdownTranslator): ...


def extract_and_format_parameters_section(content: str) -> str:
    def format_as_table(items: list[tuple[str, str, str]]) -> str:
        header = "| Name | Type | Description |\n|------|------|-------------|\n"
        rows = [
            f"| {name} | {typ} | {description} |" for name, typ, description in items
        ]
        return header + "\n".join(rows)

    pattern = r"(\* \*\*Parameters:\*\*\n((?: {2}\* .+(?:\n {4}.+)*\n?)+))"
    matches = re.findall(pattern, content)

    for match in matches:
        list_block = match[1]
        list_items = re.findall(r"( {2}\* .+(?:\n {4}.+)*)", list_block)

        extracted_items = []
        for item in list_items:
            item = item.replace("\n    ", " ")
            parts = item.split(" – ", 1)
            if len(parts) == 2:
                name_type, description = parts
            else:
                name_type = parts[0]
                description = ""

            name_match = re.search(r"\*\*(.+?)\*\*", name_type)
            type_match = re.search(r"\((.+?)\)", name_type)
            name = name_match.group(1) if name_match else ""
            typ = type_match.group(1) if type_match else ""
            typ = typ.replace("*", "").replace(" ", "").replace("|", r"\|")
            typ = f"*{typ}*"
            name = f"`{name}`"
            extracted_items.append((name, typ, description.strip()))

        table = format_as_table(extracted_items)
        content = content.replace(match[0], f"\n**Parameters:**\n\n{table}\n\n")

    return content


def _line_replacements(line: str) -> str:
    if line.startswith("### *class*"):
        line = line.replace("### *class*", "").strip()
        if not any(sym in line for sym in NON_PUBLIC_SYMBOLS):
            line = line.replace("truss_chains.definitions", "truss_chains")
        first_brace = line.find("(")
        if first_brace > 0:
            line = line[:first_brace]
        return f"### *class* `{line}`"
    elif line.startswith("### "):
        line = line.replace("### ", "").strip()
        if not any(sym in line for sym in NON_PUBLIC_SYMBOLS):
            line = line.replace("truss_chains.definitions", "truss_chains")
        first_brace = line.find("(")
        if first_brace > 0:
            line = line[:first_brace]
        return f"### `{line}`"

    return line


def _raw_text_replacements(doc_text: str) -> str:
    doc_text = doc_text.replace(PYDANTIC_DEFAULT_DOCSTRING, "")
    doc_text = doc_text.replace("Bases: `object`\n\n", "")
    doc_text = doc_text.replace("Bases: `ABCChainlet`\n\n", "")
    doc_text = doc_text.replace("Bases: `SafeModel`", "Bases: `pydantic.BaseModel`")
    doc_text = doc_text.replace(
        "Bases: `SafeModelNonSerializable`", "Bases: `pydantic.BaseModel`"
    )
    doc_text = doc_text.replace("<", "&lt;").replace(">", "&gt;")
    doc_text = "\n".join(_line_replacements(line) for line in doc_text.split("\n"))
    doc_text = extract_and_format_parameters_section(doc_text)
    return doc_text


class MDXAdapterBuilder(MarkdownBuilder):
    name = "mdx_adapter"
    out_suffix = ".mdx"
    default_translator_class = MDXAdapterTranslator

    def get_translator_class(self):
        return MDXAdapterTranslator

    def get_target_uri(self, docname: str, typ: str = None) -> str:
        return f"{docname}.mdx"

    def write_doc(self, docname: str, doctree: nodes.document):
        self.current_doc_name = docname
        self.sec_numbers = self.env.toc_secnumbers.get(docname, {})
        destination = StringOutput(encoding="utf-8")
        self.writer.write(doctree, destination)
        out_filename = os.path.join(self.outdir, f"{os_path(docname)}{self.out_suffix}")
        ensuredir(os.path.dirname(out_filename))

        with open(out_filename, "w", encoding="utf-8") as file:
            # These replacements are custom, the rest of this method is unchanged.
            file.write(_raw_text_replacements(self.writer.output))


def setup(app: Any) -> Dict[str, Any]:
    app.add_builder(MDXAdapterBuilder)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
