#!/usr/bin/env python3
import getpass
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

base_path = Path(__file__).parent.parent
templates_path = base_path / "truss" / "templates"
sys.path.append(str(base_path))


PYTHON_VERSIONS = {
    "3.7",
    "3.8",
    "3.9",
}


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
    test: bool = True,
) -> Path:
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
        # todo: refactor into function
        image_name = f"baseten/truss-{job_type}-base-{python_version}"
        if use_gpu:
            image_name = f"{image_name}-gpu"
        if live_reload:
            image_name = f"{image_name}-reload"
        tag = "latest"
        if test:
            tag = "test"
        image_with_tag = f"{image_name}:{tag}"
        print(f"Building image :: {image_with_tag}")
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


def _build_all(push: bool = False, test: bool = True):
    for job_type in ["server", "training"]:
        for python_version in ["3.9"]:
            for live_reload in [False, True]:
                for use_gpu in [False]:
                    _build(
                        job_type=job_type,
                        python_version=python_version,
                        use_gpu=use_gpu,
                        live_reload=live_reload,
                        push=push,
                        test=test,
                    )


if __name__ == "__main__":
    push_to_dockerhub = False
    if len(sys.argv) > 1 and sys.argv[1] == "push":
        push_to_dockerhub = True

    if push_to_dockerhub:
        _docker_login()

    _build_all()
