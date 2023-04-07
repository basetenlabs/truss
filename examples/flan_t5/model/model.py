import io
import os
import random
import subprocess
import tarfile
import time
from pathlib import Path
from shutil import copyfileobj
from typing import Dict, List
from urllib.request import Request, urlopen

from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._tokenizer = None
        self._model = None

    def load(self):
        tt = TimeTracker()
        data_outer_dir = Path("/tmp") / "flan_t5"
        data_dir = data_outer_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)

        install_s5cmd()

        tt.step("s5cmd install")

        data_url = "s3://baseten-dev-public/flan.tar"
        local_tar_path = "/tmp/flan.tar"
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
        self._tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        self._model = T5ForConditionalGeneration.from_pretrained(
            data_dir, device_map="auto"
        )

    def preprocess(self, request: dict):
        if "bad_words" in request:
            bad_words = request.pop("bad_words")
            # bad_words must be a list of strings, not one string
            bad_word_ids = self._tokenizer(
                bad_words, add_prefix_space=True, add_special_tokens=False
            ).input_ids
            request["bad_words_ids"] = bad_word_ids
        if "seed" in request:
            set_seed(request.pop("seed"))
        else:
            set_seed(random.randint(0, 4294967294))
        return request

    def assert_free_tier_limits(self, request: dict):
        # TODO(pankaj) Remove this
        if request.get("max_new_tokens", 0) > 100:
            raise ValueError(
                "max_tokens / max_new_tokens must be less than 101 on free tier"
            )
        if request.get("num_beams", 0) > 4:
            raise ValueError("num_beams must be less than 5 on free tier")
        if request.get("num_beam_groups", 0) > 4:
            raise ValueError("num_beam_groups must be less than 5 on free tier")

    def predict(self, request: Dict) -> Dict[str, List]:
        try:
            self.assert_free_tier_limits(request)
            decoded_output = []
            prompt = request.pop("prompt")
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(
                "cuda"
            )
            outputs = self._model.generate(input_ids, **request)
            for beam in outputs:
                decoded_output.append(
                    self._tokenizer.decode(beam, skip_special_tokens=True)
                )
        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": decoded_output, "message": None}


class TimeTracker:
    def __init__(self) -> None:
        self._ctr = time.perf_counter()

    def step(self, step_name: str):
        prev = self._ctr
        self._ctr = time.perf_counter()
        ms = int((self._ctr - prev) * 1000)
        print(f"Time taken for `{step_name}` is {ms} ms")


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
