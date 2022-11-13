from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from truss.constants import (
    REQUIREMENTS_TXT_FILENAME,
    SYSTEM_PACKAGES_TXT_FILENAME,
    TEMPLATES_DIR,
    TRAIN_DOCKERFILE_NAME,
    TRAIN_DOCKERFILE_TEMPLATE_NAME,
    TRAINING_CODE_DIR,
)
from truss.contexts.truss_context import TrussContext
from truss.docker import Docker
from truss.truss_spec import TrussSpec
from truss.utils import (
    build_truss_target_directory,
    copy_tree_path,
    given_or_temporary_dir,
)

BUILD_TRAINING_DIR_NAME = "train"


class TrainingImageBuilderContext(TrussContext):
    @staticmethod
    def run(truss_dir: Path):
        return TrainingImageBuilder(truss_dir)


class TrainingImageBuilder:
    # todo: remove duplication with image_builder, perhaps create a base class
    # or may be better to use composition by taking in build_context_preparer as input.
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
        return f"{self._spec.model_framework_name}-train:latest"

    def prepare_image_build_dir(self, build_dir: Path = None):
        """Prepare a directory for building the docker image from.

        Returns:
            docker command to build the docker image.
        """
        if build_dir is None:
            build_dir = build_truss_target_directory("train")

        copy_tree_path(self._spec.truss_dir, build_dir)

        # todo: Raise error if this ends up conflicting with truss,
        # training_model_dir can be called the same as value of TRAINING_CODE_DIR
        copy_tree_path(
            TRAINING_CODE_DIR,
            build_dir / BUILD_TRAINING_DIR_NAME,
        )

        with (build_dir / REQUIREMENTS_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.requirements_txt)

        with (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.system_packages_txt)

        bundled_packages_dir_exists = (
            build_dir / self._spec.config.bundled_packages_dir
        ).exists()
        template_loader = FileSystemLoader(str(TEMPLATES_DIR))
        template_env = Environment(loader=template_loader)
        dockerfile_template = template_env.get_template(TRAIN_DOCKERFILE_TEMPLATE_NAME)
        dockerfile_contents = dockerfile_template.render(
            config=self._spec.config,
            bundled_packages_dir_exists=bundled_packages_dir_exists,
        )
        docker_file_path = build_dir / TRAIN_DOCKERFILE_NAME
        with docker_file_path.open("w") as docker_file:
            docker_file.write(dockerfile_contents)

    def docker_build_command(self, build_dir) -> str:
        return f"docker build {build_dir} -t {self.default_tag}"
