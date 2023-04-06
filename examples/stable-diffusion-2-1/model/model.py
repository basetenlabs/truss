import base64
import io
import os
import shutil
import subprocess
import tarfile
import time
from io import BytesIO
from pathlib import Path
from shutil import copyfileobj
from typing import Dict, List
from urllib.request import Request, urlopen

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def install_s5cmd(install_path="/usr/local/bin"):
    if (Path(install_path) / "s5cmd").exists():
        return

    url = "https://github.com/peak/s5cmd/releases/download/v2.1.0-beta.1/s5cmd_2.1.0-beta.1_Linux-64bit.tar.gz"
    # Download s5cmd binary
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    binary_tar = io.BytesIO(urlopen(req).read())

    # Extract the binary
    with tarfile.open(fileobj=binary_tar, mode="r:gz") as tar:
        s5cmd_binary = tar.extractfile("s5cmd")

        # Save the binary to the install path
        with open(os.path.join(install_path, "s5cmd"), "wb") as outfile:
            copyfileobj(s5cmd_binary, outfile)
            os.chmod(outfile.name, 0o755)


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs.get("secrets")
        self._model = None
        self._scheduler = None

    def load(self):
        # Download data dir
        tt = TimeTracker()
        data_outer_dir = Path("/cache") / "sd"
        data_dir = data_outer_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)

        install_s5cmd()

        tt.step("s5cmd install")

        data_url = "s3://baseten-dev-public/sd21.tar"
        local_tar_path = "/cache/sd21.tar"
        subprocess.run(
            [
                "s5cmd",
                "--numworkers",
                "10",
                "--no-sign-request",
                "cp",
                data_url,
                local_tar_path,
            ],
            check=False,
        )
        tt.step("model data download")
        with tarfile.open(local_tar_path) as data_tar:
            data_tar.extractall(str(data_outer_dir))
            Path(local_tar_path).unlink()
        tt.step("tar extraction")
        self._scheduler = EulerDiscreteScheduler.from_pretrained(
            data_dir, subfolder="scheduler"
        )
        tt.step("scheduler load")
        self._model = StableDiffusionPipeline.from_pretrained(
            data_dir, scheduler=self._scheduler, torch_dtype=torch.float16
        )

        tt.step("model load to memory")
        self._model = self._model.to("cuda")
        tt.step("model load to GPU")
        shutil.rmtree(str(data_outer_dir))
        tt.step("rmtree")
        self._model.enable_xformers_memory_efficient_attention()

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    @torch.inference_mode()
    def predict(self, request: Dict) -> Dict[str, List]:
        prompt = request.pop("prompt")
        results = []
        try:
            output = self._model(prompt=prompt, return_dict=False, **request)

            for image in output[0]:
                b64_results = self.convert_to_b64(image)
                results.append(b64_results)

        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": results, "message": None}


class TimeTracker:
    def __init__(self) -> None:
        self._ctr = time.perf_counter()

    def step(self, step_name: str):
        prev = self._ctr
        self._ctr = time.perf_counter()
        ms = int((self._ctr - prev) * 1000)
        print(f"Time taken for `{step_name}` is {ms} ms")
