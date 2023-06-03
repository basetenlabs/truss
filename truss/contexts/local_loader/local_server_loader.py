#!/usr/bin/env python3
import subprocess
import sys
import venv
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

import dockerfile
from truss.contexts.image_builder.serving_image_builder import ServingImageBuilder
from truss.util.path import (
    build_truss_target_directory,
    copy_file_path,
    copy_tree_or_file,
)
from watchfiles import Change, watch


class EnvBuilder(venv.EnvBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = None

    def post_setup(self, context):
        self.context = context


class DockerBuildEmulator:
    def __init__(self, dockerfile_path: Path) -> None:
        self.commands = dockerfile.parse_file(str(dockerfile_path))
        self.env_vars: Dict[str, str] = {}
        self.entry_point: List[str] = []
        self.work_dir: Path = Path("/")

    def _resolve_env(self, key: str) -> str:
        if key.startswith("$"):
            key = key.replace("$", "", 1)
            v = self.env_vars[key]
            return v
        return key

    def _resolve_values(self, keys: List[str]) -> List[str]:
        return list(map(self._resolve_env, keys))

    def run(self, context_dir: Path, fs_root_dir: Path):
        for cmd in self.commands:
            values = self._resolve_values(cmd.value)
            if cmd.cmd == "ENV":
                self.env_vars[values[0]] = values[1]
            if cmd.cmd == "ENTRYPOINT":
                self.entry_point = list(values)
            if cmd.cmd == "COPY":
                # symlink to path
                src, dst = values
                src = src.replace("./", "", 1)
                dst = dst.replace("/", "", 1)
                copy_tree_or_file(context_dir / src, fs_root_dir / dst)
            if cmd.cmd == "WORKDIR":
                self.work_dir = self.work_dir / values[0]


class TrussFilesWatcher(Thread):
    def __init__(self, watch_path: Path, mirror_path: Path) -> None:
        super().__init__()
        self.watch_path = watch_path
        self.mirror_path = mirror_path

    def run(self):
        for changes in watch(str(self.watch_path)):
            for change in changes:
                op, path = change
                rel_path = Path(path).relative_to(self.watch_path.resolve())
                if op == Change.modified:
                    print(rel_path)
                    copy_file_path(
                        self.watch_path / rel_path, self.mirror_path / rel_path
                    )
            #  print(path)
            #  print(self.watch_path.resolve())
            #  print(op, )
            # print(changes)

    def stop(self):
        self._stop()
        self.join()


class LocalServerLoader:
    def __init__(self, truss_path: Path, context_builder: ServingImageBuilder) -> None:
        self.truss_path = truss_path
        self.context_builder = context_builder

    def watch(self, build_dir: Optional[Path] = None, venv_dir: Optional[Path] = None):
        if build_dir is None:
            build_dir = build_truss_target_directory("build_dir")
        else:
            if not build_dir.exists():
                build_dir.mkdir(parents=True)

        if venv_dir is None:
            venv_dir = build_truss_target_directory("venv")
        else:
            if not venv_dir.exists():
                venv_dir.mkdir(parents=True)

        self.context_builder.prepare_image_build_dir(build_dir)
        dockerfile_path = build_dir / "Dockerfile"
        docker_build = DockerBuildEmulator(dockerfile_path)
        docker_build.run(build_dir, venv_dir)

        execution_env_vars = docker_build.env_vars

        # print(f" *** Created temporary directory '{target_dir_path}'.")
        venv_builder = EnvBuilder(with_pip=True)
        venv_builder.create(str(venv_dir / ".env"))
        venv_context = venv_builder.context

        requirements_files = [
            "app/requirements.txt",
            "requirements.txt",
        ]
        for req_file in requirements_files:
            req_file_path = venv_dir / req_file

            if req_file_path.exists():
                print(f"Installing requirements for {req_file_path}")
                pip_install_command = [
                    venv_context.env_exe,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(req_file_path.absolute()),
                ]
                subprocess.check_call(pip_install_command)

        # Modify path to run in proper root
        venv_entry_point = [
            venv_context.env_exe,
            "-m",
            "uvicorn",
            "local_inference_server:app",
            "--reload",
        ]
        t = TrussFilesWatcher(self.truss_path, venv_dir / "app/")
        t.start()
        execution_env_vars = {
            **execution_env_vars,
            "RELOAD": "True",
            "RELOAD_DIRS": str(venv_dir.resolve()),
        }

        subprocess.check_call(
            venv_entry_point,
            cwd=str(venv_dir / str(docker_build.work_dir).replace("/", "", 1)),
            env=execution_env_vars,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        t.stop()
