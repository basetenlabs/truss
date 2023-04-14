from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


def read_template_from_fs(base_dir: Path, template_file_name: str) -> Template:
    template_loader = FileSystemLoader(str(base_dir))
    template_env = Environment(loader=template_loader)
    return template_env.get_template(template_file_name)
