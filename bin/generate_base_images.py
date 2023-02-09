#!/usr/bin/env python3
import getpass
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader
from truss.contexts.image_builder.util import (
    truss_base_image_name,
    truss_base_image_tag,
)

base_path = Path(__file__).parent.parent
templates_path = base_path / "truss" / "templates"
sys.path.append(str(base_path))


PYTHON_VERSIONS = {
    "3.8",
    "3.9",
}


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


def _render_dockerfile(
    job_type: str,
    python_version: str,
    live_reload: bool,
    use_gpu: bool,
) -> str:
    # Render jinja
    jinja_env = Environment(
        loader=FileSystemLoader(str(base_path / "docker" / "base_images")),
    )
    template = jinja_env.get_template("base_image.Dockerfile.jinja")
    return template.render(
        use_gpu=use_gpu,
        live_reload=live_reload,
        job_type=job_type,
        python_version=python_version,
    )


def _build(
    python_version: str,
    live_reload: bool = False,
    use_gpu: bool = False,
    job_type: str = "server",
    push: bool = False,
    version_tag: str = None,
    dry_run: bool = True,
):
    image_name = truss_base_image_name(job_type=job_type)
    tag = truss_base_image_tag(
        python_version=python_version,
        use_gpu=use_gpu,
        live_reload=live_reload,
        version_tag=version_tag,
    )
    image_with_tag = f"{image_name}:{tag}"
    print(f"Building image :: {image_with_tag}")
    if dry_run:
        return

    dockerfile_content = _render_dockerfile(
        job_type=job_type,
        python_version=python_version,
        live_reload=live_reload,
        use_gpu=use_gpu,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        build_ctx_path = Path(temp_dir)
        with (build_ctx_path / "Dockerfile").open("w") as dockerfile_file:
            dockerfile_file.write(dockerfile_content)

        if job_type == "server":
            reqs_copy_from = templates_path / "server" / "requirements.txt"
        elif job_type == "training":
            reqs_copy_from = templates_path / "training" / "requirements.txt"
        else:
            raise ValueError(f"Unknown job type {job_type}")

        shutil.copyfile(
            str(reqs_copy_from),
            str(build_ctx_path / "requirements.txt"),
        )
        shutil.copytree(
            str(templates_path / "control"),
            str(build_ctx_path / "control"),
        )
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform=linux/amd64",
            ".",
            "-t",
            image_with_tag,
        ]
        if push:
            cmd.append("--push")
        subprocess.run(cmd, cwd=build_ctx_path)


def _build_all(
    job_types: List[str] = None,
    python_versions: List[str] = None,
    live_reload_values: List[bool] = None,
    use_gpu_values: List[bool] = None,
    push: bool = False,
    version_tag: str = None,
    dry_run: bool = False,
):
    if job_types is None:
        job_types = ["server", "training"]

    if python_versions is None:
        python_versions = PYTHON_VERSIONS

    if live_reload_values is None:
        live_reload_values = [True, False]

    if use_gpu_values is None:
        use_gpu_values = [True, False]

    for job_type in job_types:
        for python_version in python_versions:
            for live_reload in live_reload_values:
                for use_gpu in use_gpu_values:
                    _build(
                        job_type=job_type,
                        python_version=python_version,
                        use_gpu=use_gpu,
                        live_reload=live_reload,
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
        "--job-type",
        nargs="?",
        default="all",
        choices=["server", "training", "all"],
        help="Create images for server",
    )
    parser.add_argument(
        "--live-reload",
        nargs="?",
        default="both",
        choices=["y", "n", "both"],
        help="Whether to create live-reload capable, incapable or both images",
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
        choices=[*PYTHON_VERSIONS, "all"],
        help="Build images for specific python version or all",
    )
    parser.add_argument(
        "--skip-login",
        action=BooleanOptionalAction,
        default=False,
        help="Skip docker login even if push is specire",
    )

    args = parser.parse_args()
    if args.python_version == "all":
        python_versions = PYTHON_VERSIONS
    else:
        python_versions = [args.python_version]

    if args.job_type == "all":
        job_types = ["server", "training"]
    else:
        job_types = [args.job_type]

    use_gpu_values = _bool_arg_str_to_values(args.use_gpu)
    live_reload_values = _bool_arg_str_to_values(args.live_reload)

    if args.push and not args.skip_login:
        _docker_login()

    _build_all(
        python_versions=python_versions,
        job_types=job_types,
        use_gpu_values=use_gpu_values,
        live_reload_values=live_reload_values,
        push=args.push,
        version_tag=args.version_tag,
        dry_run=args.dry_run,
    )
