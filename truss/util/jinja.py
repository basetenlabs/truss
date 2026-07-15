from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


def dockerfile_env_value(value: str) -> str:
    """Quote a string as a Dockerfile ENV value. Unlike `tojson`, whose
    \\uXXXX escapes for ' < > & the ENV parser keeps verbatim.

    Escapes exactly the special set of buildkit's double-quoted ENV grammar,
    {" $ \\} (shell/lex.go processDoubleQuote); all else passes verbatim."""
    if "\n" in value:
        raise ValueError(f"Dockerfile ENV values cannot contain newlines: {value!r}")
    # Escaping $ defers env expansion from image build time to runtime.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return f'"{escaped}"'


def read_template_from_fs(base_dir: Path, template_file_name: str) -> Template:
    template_loader = FileSystemLoader(str(base_dir))
    template_env = Environment(loader=template_loader)
    template_env.filters["dockerfile_env_value"] = dockerfile_env_value
    return template_env.get_template(template_file_name)
