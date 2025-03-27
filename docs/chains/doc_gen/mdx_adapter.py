# type: ignore
"""Super hacky plugin to make the generated markdown more suitable for
rendering in mintlify as an mdx doc."""

import importlib
import inspect
import os
import re
from typing import Any, Dict

import pydantic
from docutils import nodes
from docutils.io import StringOutput
from pydantic.fields import PydanticUndefined
from sphinx.util.osutil import ensuredir, os_path
from sphinx_markdown_builder import MarkdownBuilder
from sphinx_markdown_builder.translator import MarkdownTranslator
from tabulate import tabulate

PYDANTIC_DEFAULT_DOCSTRING = """

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

self is explicitly positional-only to allow self as a field name.
"""


class MDXAdapterTranslator(MarkdownTranslator):
    pass


def _split_sections_by_heading(content: str) -> list[str]:
    return re.split(r"(?=\n### )", content)


def _extract_section_header(section: str) -> tuple[str, str] | None:
    match = re.match(r"\n?### \*(\w+)\*\s+`([\w\.\:]+)`", section)
    if match:
        return (match.group(1), match.group(2))
    match = re.match(r"\n?### `([\w\.\:]+)`", section)
    if match:
        return ("function", match.group(1))
    return None


def _format_default(obj: Any) -> str:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return repr(obj)
    if isinstance(obj, (list, dict, set, tuple)):
        try:
            return repr(obj)
        except Exception:
            pass
    cls = obj.__class__
    cls_md = f"{cls.__module__}.{cls.__qualname__}()"
    return cls_md.replace("truss_chains.public_types", "truss_chains")


def _get_model_field_defaults(full_class_path: str) -> dict[str, Any]:
    print(f"Attempting to load defaults from model: {full_class_path}")
    try:
        module_path, cls_name = full_class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        model_cls = getattr(mod, cls_name)
        if issubclass(model_cls, pydantic.BaseModel):
            defaults = {}
            for name, field in model_cls.model_fields.items():
                if field.default is not PydanticUndefined:
                    value = field.default
                elif field.default_factory is not None:
                    value = field.default_factory()
                else:
                    continue

                print(f"@@@ {name}: {value}")
                defaults[name] = _format_default(value)

            return defaults
    except Exception as e:
        print(f"Could not load model defaults for {full_class_path}: {e}")
    return {}


def _get_function_defaults(full_func_path: str) -> dict[str, Any]:
    # For methods or functions, parse signature to extract defaults.
    print(f"Attempting to load defaults from function: {full_func_path}")
    try:
        module_path, func_name = full_func_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        sig = inspect.signature(func)
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param.default is not param.empty:
                defaults[param_name] = _format_default(param.default)
        return defaults
    except Exception as e:
        print(f"Could not load function defaults for {full_func_path}: {e}")
    return {}


def _format_and_inject_parameters(section: str, field_defaults: dict[str, Any]) -> str:
    pattern = r"(\* \*\*Parameters:\*\*\n((?: {2}\* .+(?:\n {4}.+)*\n?)+))"
    matches = re.findall(pattern, section)

    for full_match, list_block in matches:
        list_items = re.findall(r"( {2}\* .+(?:\n {4}.+)*)", list_block)
        extracted_items = []
        for item in list_items:
            parsed = _parse_param_item(item, field_defaults)
            print(parsed)
            extracted_items.append(parsed)

        table = _format_as_table(extracted_items)
        # Replace the entire parameters bullet block with a new table
        section = section.replace(full_match, f"\n**Parameters:**\n\n{table}\n\n")

    return section


def _parse_param_item(
    item: str, field_defaults: dict[str, Any]
) -> tuple[str, str, str, str]:
    # parse lines like:
    #   "  * **param_name** (str) – Description of param."
    item = item.replace("\n    ", " ")
    parts = item.split(" – ", 1)
    name_type = parts[0]
    description = parts[1] if len(parts) == 2 else ""

    name_match = re.search(r"\*\*(.+?)\*\*", name_type)
    type_match = re.search(r"\((.+?)\)", name_type)

    name = name_match.group(1) if name_match else ""
    typ = type_match.group(1) if type_match else ""
    # handle table-breaking chars
    typ = typ.replace("*", "").replace(" ", "").replace("|", r"\|")

    # final formatting
    param_name_md = f"`{name}`"
    param_type_md = f"*{typ}*"

    default_val = field_defaults.get(name, "")
    default_val_md = ""
    if default_val != "":
        default_val_md = f"`{default_val}`"
    return (param_name_md, param_type_md, default_val_md, description.strip())


def _format_as_table(items: list[tuple[str, str, str, str]]) -> str:
    headers = ["Name", "Type", "Default", "Description"]
    if all(i[2] == "" for i in items):
        headers = ["Name", "Type", "Description"]
        items = [(i[0], i[1], i[2]) for i in items]

    return tabulate(items, headers=headers, tablefmt="github")


def extract_and_format_parameters_section(content: str) -> str:
    sections = _split_sections_by_heading(content)
    updated_sections = []

    for section in sections:
        print(f"Processing section: `{section[:80]}...`")
        kind, full_name = _extract_section_header(section) or (None, None)
        print(f"kind=`{kind}`, full_name=`{full_name}`")
        field_defaults: dict[str, Any] = {}

        if kind == "class" and full_name:
            field_defaults = _get_model_field_defaults(full_name)
        elif kind in ("function", "method") and full_name:
            field_defaults = _get_function_defaults(full_name)

        print(f"field_defaults={field_defaults}")
        section = _format_and_inject_parameters(section, field_defaults)
        updated_sections.append(section)

    return "".join(updated_sections)


def _line_replacements(line: str) -> str:
    if line.startswith("### *class*"):
        line = line.replace("### *class*", "").strip()
        first_brace = line.find("(")
        if first_brace > 0:
            line = line[:first_brace]
        return f"\n### *class* `{line}`"
    elif line.startswith("### *function*"):
        line = line.replace("### *function*", "").strip()
        first_brace = line.find("(")
        if first_brace > 0:
            line = line[:first_brace]
        return f"\n### *function* `{line}`"
    elif line.startswith("### "):
        # generic fallback for headings
        line = line.replace("### ", "").strip()
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

    lines = doc_text.split("\n")
    new_lines = []
    for line in lines:
        new_lines.append(_line_replacements(line))
    doc_text = "\n".join(new_lines)

    doc_text = extract_and_format_parameters_section(doc_text)
    return doc_text


AUTOGEN_NOTE = """{/*
This file is autogenerated, do not edit manually, see:
https://github.com/basetenlabs/truss/tree/main/docs/chains/doc_gen
*/}
"""


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
            content = _raw_text_replacements(self.writer.output)
            file.write(AUTOGEN_NOTE + "\n" + content)


def setup(app: Any) -> Dict[str, Any]:
    app.add_builder(MDXAdapterBuilder)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
