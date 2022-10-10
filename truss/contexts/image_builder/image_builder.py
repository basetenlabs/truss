from pathlib import Path

import click
from jinja2 import Template
from truss.constants import (
    CONTROL_SERVER_CODE_DIR,
    MODEL_DOCKERFILE_NAME,
    MODEL_README_NAME,
    REQUIREMENTS_TXT_FILENAME,
    SERVER_CODE_DIR,
    SERVER_DOCKERFILE_TEMPLATE_NAME,
    SERVER_REQUIREMENTS_TXT_FILENAME,
    SYSTEM_PACKAGES_TXT_FILENAME,
    TEMPLATES_DIR,
)
from truss.contexts.truss_context import TrussContext
from truss.docker import Docker
from truss.patch.hash import directory_content_hash
from truss.readme_generator import generate_readme
from truss.truss_spec import TrussSpec
from truss.utils import (
    build_truss_target_directory,
    copy_file_path,
    copy_tree_path,
    given_or_temporary_dir,
)

BUILD_SERVER_DIR_NAME = "server"
BUILD_CONTROL_SERVER_DIR_NAME = "control"


class ImageBuilderContext(TrussContext):
    @staticmethod
    def run(truss_dir: Path):
        return ImageBuilder(truss_dir)


class ImageBuilder:
    def __init__(self, truss_dir: Path) -> None:
        self._truss_dir = truss_dir
        self._spec = TrussSpec(truss_dir)

    def build_image(self, build_dir: Path = None, tag: str = None, labels: dict = None):
        """Build image.

        Arguments:
            build_dir(Path): Directory to use for building the docker image. If None
                             then a temporary directory is used.
            tag(str): A tag to assign to the docker image.
        """

        with given_or_temporary_dir(build_dir) as build_dir_path:
            self.prepare_image_build_dir(build_dir_path)
            return Docker.client().build(
                str(build_dir_path),
                labels=labels if labels else {},
                tags=tag or self.default_tag,
            )

    @property
    def default_tag(self):
        return f"{self._spec.model_framework_name}-model:latest"

    def prepare_image_build_dir(self, build_dir: Path = None):
        """Prepare a directory for building the docker image from.

        Returns:
            docker command to build the docker image.
        """
        if build_dir is None:
            build_dir = build_truss_target_directory(self._spec.model_framework_name)
            # todo: Add a logging statement here, suggesting how to clean up the directory.

        copy_tree_path(self._spec.truss_dir, build_dir)
        copy_tree_path(
            SERVER_CODE_DIR,
            build_dir / BUILD_SERVER_DIR_NAME,
        )
        if self._spec.config.use_control_plane:
            copy_tree_path(
                CONTROL_SERVER_CODE_DIR,
                build_dir / BUILD_CONTROL_SERVER_DIR_NAME,
            )
        copy_file_path(
            TEMPLATES_DIR / self._spec.model_framework_name / REQUIREMENTS_TXT_FILENAME,
            build_dir / SERVER_REQUIREMENTS_TXT_FILENAME,
        )

        with (build_dir / REQUIREMENTS_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.requirements_txt)

        with (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.system_packages_txt)

        dockerfile_template_path = TEMPLATES_DIR / SERVER_DOCKERFILE_TEMPLATE_NAME

        with dockerfile_template_path.open() as dockerfile_template_file:
            dockerfile_template = Template(dockerfile_template_file.read())
            data_dir_exists = (build_dir / self._spec.config.data_dir).exists()
            bundled_packages_dir_exists = (
                build_dir / self._spec.config.bundled_packages_dir
            ).exists()
            dockerfile_contents = dockerfile_template.render(
                config=self._spec.config,
                data_dir_exists=data_dir_exists,
                bundled_packages_dir_exists=bundled_packages_dir_exists,
                truss_hash=directory_content_hash(self._truss_dir),
            )
            docker_file_path = build_dir / MODEL_DOCKERFILE_NAME
            with docker_file_path.open("w") as docker_file:
                docker_file.write(dockerfile_contents)

        readme_file_path = build_dir / MODEL_README_NAME
        try:
            readme_contents = generate_readme(self._spec)
            with readme_file_path.open("w") as readme_file:
                readme_file.write(readme_contents)
        except Exception as e:
            click.echo(
                click.style(
                    f"""WARNING: Auto-readme generation has failed.
                    This is probably due to a malformed config.yaml or
                    malformed examples.yaml. Error is:
                    {e}
                    """,
                    fg="yellow",
                )
            )

    def docker_build_command(self, build_dir) -> str:
        return f"docker build {build_dir} -t {self.default_tag}"
