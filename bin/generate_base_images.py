#!/usr/bin/env python3
import getpass
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction  # type: ignore
from pathlib import Path
from typing import Iterable, List, Optional

from jinja2 import Environment, FileSystemLoader

from truss.base.constants import SUPPORTED_PYTHON_VERSIONS, TRUSS_BASE_IMAGE_NAME
from truss.contexts.image_builder.util import truss_base_image_tag

base_path = Path(__file__).parent.parent
templates_path = base_path / "truss" / "templates"
sys.path.append(str(base_path))


def _bool_arg_str_to_values(bool_arg_str: str) -> List[bool]:
    if bool_arg_str == "both":
        return [True, False]
    if bool_arg_str == "y":
        return [True]
    if bool_arg_str == "n":
        return [False]
    raise ValueError(
        "Unexpected str value of bool flag, acceptable values are y/n/both"
    )


def _docker_login():
    pw = getpass.getpass(
        prompt="Please supply password for `basetenservice` dockerhub account: "
    )
    subprocess.run(["docker", "login", "-u", "basetenservice", "-p", pw])


def _render_dockerfile(python_version: str, use_gpu: bool) -> str:
    # Render jinja
    jinja_env = Environment(
        loader=FileSystemLoader(str(base_path / "docker" / "base_images"))
    )
    template = jinja_env.get_template("base_image.Dockerfile.jinja")
    return template.render(use_gpu=use_gpu, python_version=python_version)


def _build(
    python_version: str,
    use_gpu: bool = False,
    push: bool = False,
    version_tag: Optional[str] = None,
    dry_run: bool = True,
):
    tag = truss_base_image_tag(
        python_version=python_version, use_gpu=use_gpu, version_tag=version_tag
    )
    image_with_tag = f"{TRUSS_BASE_IMAGE_NAME}:{tag}"
    print(f"Building image :: {image_with_tag}")
    if dry_run:
        return

    dockerfile_content = _render_dockerfile(
        python_version=python_version, use_gpu=use_gpu
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        build_ctx_path = Path(temp_dir)
        with (build_ctx_path / "Dockerfile").open("w") as dockerfile_file:
            dockerfile_file.write(dockerfile_content)

        reqs_copy_from = templates_path / "server" / "requirements.txt"
        shutil.copyfile(str(reqs_copy_from), str(build_ctx_path / "requirements.txt"))
        shutil.copytree(
            str(templates_path / "control"), str(build_ctx_path / "control")
        )
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform=linux/arm64,linux/amd64",
            ".",
            "-t",
            image_with_tag,
        ]
        if push:
            cmd.append("--push")

        # Needed to support multi-arch build.
        subprocess.run(
            ["docker", "buildx", "create", "--use"], cwd=build_ctx_path, check=True
        )
        subprocess.run(cmd, cwd=build_ctx_path, check=True)


def _build_all(
    python_versions: Iterable[str] = SUPPORTED_PYTHON_VERSIONS,
    use_gpu_values: Iterable[bool] = (True, False),
    push: bool = False,
    version_tag: Optional[str] = None,
    dry_run: bool = False,
):
    for python_version in python_versions:
        for use_gpu in use_gpu_values:
            _build(
                python_version=python_version,
                use_gpu=use_gpu,
                push=push,
                version_tag=version_tag,
                dry_run=dry_run,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Publish truss base images")
    parser.add_argument(
        "--push",
        action=BooleanOptionalAction,
        default=False,
        help="push built images to dockerhub",
    )
    parser.add_argument(
        "--version-tag",
        nargs="?",
        help="Generate images with given version tag, useful for testing. "
        "If absent then truss project version is used.",
    )
    parser.add_argument(
        "--dry-run",
        action=BooleanOptionalAction,
        default=False,
        help="Print only the image names that will be built",
    )
    parser.add_argument(
        "--use-gpu",
        nargs="?",
        default="both",
        choices=["y", "n", "both"],
        help="Whether to create gpu capable, incapable or both images",
    )
    parser.add_argument(
        "--python-version",
        nargs="?",
        default="all",
        choices=[*SUPPORTED_PYTHON_VERSIONS, "all"],
        help="Build images for specific python version or all",
    )
    parser.add_argument(
        "--skip-login",
        action=BooleanOptionalAction,
        default=False,
        help="Skip docker login even if push is specified.",
    )

    args = parser.parse_args()
    if args.python_version == "all":
        python_versions = SUPPORTED_PYTHON_VERSIONS
    else:
        python_versions = [args.python_version]

    use_gpu_values = _bool_arg_str_to_values(args.use_gpu)

    if args.push and not args.skip_login:
        _docker_login()

    _build_all(
        python_versions=python_versions,
        use_gpu_values=use_gpu_values,
        push=args.push,
        version_tag=args.version_tag,
        dry_run=args.dry_run,
    )
