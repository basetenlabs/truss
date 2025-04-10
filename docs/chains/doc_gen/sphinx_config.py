import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("."))


project = "Dummy"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_markdown_builder",
    # "sphinx_markdown_parser",
    "sphinx-pydantic",
    # "myst_parser",
    "mdx_adapter",
]
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "exclude-members": "__*",
    "inherited-members": False,
    "show-inheritance": True,
}


# Other Options.
autodoc_typehints = "description"
always_document_param_types = True
# Include both class-level and __init__ docstrings
autoclass_content = "both"
# Napoleon (docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
# HTML output.
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


def skip_member(app, what, name, obj, skip, options):
    if name == "Config" and isinstance(obj, type):
        # print(options.parent)
        return True
    # Exclude Pydantic's Config class and internal attributes
    pydantic_internal_attributes = {
        "model_computed_fields",
        "model_fields",
        "model_json_schema",
        "model_config",
    }
    if name in pydantic_internal_attributes:
        # This shadows user defined usage of those names...
        return True
    return skip


def dump_doctree(app, doctree, docname: str):
    output_file = f"/tmp/doc_gen/doctree_{docname}.txt"  # Define the output file name

    def visit_node(node, depth=0, file=sys.stdout):
        # Create a visual guide with indentation and vertical lines
        indent = "│   " * depth + "├── "
        newl = "\n"
        node_text = node.astext()[:100].replace(newl, " ")
        file.write(
            f"{indent}{node.__class__.__name__}: `{node_text}`.\n"
        )  # Write to file

        if not node.children:  # Check if the node is a leaf node
            empty_indent = "│   " * depth
            file.write(f"{empty_indent}\n")

        for child in node.children:
            visit_node(child, depth + 1, file)

    with open(output_file, "w") as file:  # Open the file for writing
        file.write(f"Dumping doctree for: {docname}\n")
        visit_node(
            doctree, file=file
        )  # Pass the file handle to the visit_node function
        file.write("\nFinished dumping doctree\n")


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    # app.connect("doctree-resolved", dump_doctree)
