import json
import shlex
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


def dockerfile_env_value(value: str) -> str:
    """Quote a string as a Dockerfile ENV value. Unlike `tojson`, whose
    \\uXXXX escapes for ' < > & the ENV parser keeps verbatim.

    Escapes exactly the special set of buildkit's double-quoted ENV grammar,
    {" $ \\} (shell/lex.go processDoubleQuote); all else passes verbatim."""
    if "\n" in value or "\r" in value:
        # Deliberately not echoing the value: start commands can embed secrets.
        raise ValueError("Dockerfile ENV values cannot contain line breaks.")
    # Escaping $ defers env expansion from image build time to runtime.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return f'"{escaped}"'


def dockerfile_shell_value(value: str) -> str:
    """Quote a string as one POSIX shell word in a Dockerfile RUN.

    `dockerfile_env_value` follows buildkit ENV grammar. RUN still goes
    through /bin/sh, so backticks and other shell substitutions remain live
    there. `shlex.quote` uses single quotes, which stops that.
    """
    if "\n" in value or "\r" in value:
        # Deliberately not echoing the value: URLs and paths can embed secrets.
        raise ValueError("Dockerfile RUN values cannot contain line breaks.")
    return shlex.quote(value)


def dockerfile_copy_path(value: str) -> str:
    """Quote a path as one element of a JSON-form Dockerfile COPY.

    The JSON form keeps whitespace and `<<` in paths literal. Buildkit still
    runs each decoded path through word expansion, which treats \\ $ ' and "
    specially, so those are backslash-escaped before JSON encoding.
    """
    escaped = "".join(f"\\{c}" if c in "\\$'\"" else c for c in value)
    return json.dumps(escaped)


def read_template_from_fs(base_dir: Path, template_file_name: str) -> Template:
    template_loader = FileSystemLoader(str(base_dir))
    template_env = Environment(loader=template_loader)
    template_env.filters["dockerfile_env_value"] = dockerfile_env_value
    template_env.filters["dockerfile_shell_value"] = dockerfile_shell_value
    template_env.filters["dockerfile_copy_path"] = dockerfile_copy_path
    return template_env.get_template(template_file_name)
