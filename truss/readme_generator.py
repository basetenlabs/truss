from jinja2 import Template
from truss.constants import README_TEMPLATE_NAME, TEMPLATES_DIR
from truss.truss_spec import TrussSpec


def generate_readme(_spec: TrussSpec) -> str:
    readme_template_path = TEMPLATES_DIR / README_TEMPLATE_NAME
    with readme_template_path.open() as readme_template_file:
        readme_template = Template(readme_template_file.read())
        # examples.yaml may not exist
        # if examples.yaml does exist, but it's empty, examples_raw is None
        examples_raw = _spec.examples if _spec.examples_path.exists() else None
        readme_contents = readme_template.render(
            config=_spec.config, examples=examples_raw
        )
    return readme_contents
