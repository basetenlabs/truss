#!/usr/bin/env python3
import getpass
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from truss.model_framework import ModelFramework
from truss.model_frameworks import MODEL_FRAMEWORKS_BY_TYPE, SUPPORTED_MODEL_FRAMEWORKS

base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

from truss.model_inference import PYTHON_VERSIONS
from truss.types import ModelFrameworkType


def _docker_login():
    pw = getpass.getpass(
        prompt="Please supply password for `basetenservice` dockerhub account: "
    )
    subprocess.run(["docker", "login", "-u", "basetenservice", "-p", pw])


def _render_dockerfile(
    model_framework: str,
    python_version: str,
    live_reload: bool,
    use_gpu: bool,
) -> str:
    # Render jinja
    jinja_env = Environment(
        loader=FileSystemLoader(str(base_path / 'docker' / 'base_images')),
    )
    template = jinja_env.get_template('base_image.Dockerfile.jinja')
    return template.render({
        'use_gpu': use_gpu,
        'live_reload': live_reload,
        'model_framework': model_framework,
        'python_version': python_version,
    })


def _build(
    model_framework: ModelFramework,
    python_version: str,
    live_reload: bool = False,
    use_gpu: bool = False,
    push: bool = False,
    test: bool = True,
) -> Path:
    """Builds docker image."""
    model_framework_name = model_framework.typ().value
    dockerfile_content = _render_dockerfile(
        model_framework=model_framework_name,
        python_version=python_version,
        live_reload=live_reload,
        use_gpu=use_gpu,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        build_ctx_path = Path(temp_dir)
        with (build_ctx_path / 'Dockerfile').open('w') as dockerfile_file:
            dockerfile_file.write(dockerfile_content)
        server_requirements_path = build_ctx_path / 'server_requirements.txt'
        shutil.copyfile(
            str(base_path / 'truss' / 'templates' / 'server' / 'requirements.txt'),
            str(server_requirements_path),
        )
        with server_requirements_path.open('a') as reqs_file:
            for req in model_framework.requirements_txt():
                reqs_file.write('\n')
                reqs_file.write(req)
        shutil.copytree(
            str(base_path / 'truss' / 'templates' / 'control'),
            str(build_ctx_path / 'control'),
        )
        # todo: refactor into function
        image_name = f"baseten/truss-base-{python_version}-{model_framework_name}"
        if use_gpu:
            image_name = f'{image_name}-gpu'
        if live_reload:
            image_name = f'{image_name}-reload'
        tag = 'latest'
        if test:
            tag = 'test'

        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform=linux/amd64",
            ".",
            "-t",
            f"{image_name}:{tag}",
        ]
        if push:
            cmd.append("--push")
        subprocess.run(cmd, cwd=build_ctx_path)


def _build_all_for_model_framework(
    model_framework: ModelFramework,
    push: bool = False,
    test: bool = True,
):
    for python_version in PYTHON_VERSIONS:
        for live_reload in [True, False]:
            for use_gpu in [True, False]:
                _build(
                    model_framework=model_framework,
                    python_version=python_version,
                    use_gpu=use_gpu,
                    live_reload=live_reload,
                    push=push,
                    test=test,
                )


def _build_all(push: bool = False, test: bool = True):
    for model_framework in MODEL_FRAMEWORKS_BY_TYPE.values():
        _build_all_for_model_framework(model_framework, push, test)


if __name__ == '__main__':
    # _docker_login()
    # _build(
    #     model_framework=MODEL_FRAMEWORKS_BY_TYPE[ModelFrameworkType.CUSTOM],
    #     python_version="py39",
    #     use_gpu=True,
    # )
    push = False
    if len(sys.argv) > 1 and sys.argv[1] == 'push':
        push = True

    if push:
        _docker_login()

    _build_all_for_model_framework(MODEL_FRAMEWORKS_BY_TYPE[ModelFrameworkType.CUSTOM])
