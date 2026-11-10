"""Launch BEIv3 using the prepared argv and runtime-mounted HF credentials."""

import json
import os
from pathlib import Path


def main() -> None:
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    environment = os.environ.copy()
    if environment.get("HF_TOKEN") and not environment.get("BEI_HF_TOKEN"):
        environment["BEI_HF_TOKEN"] = environment["HF_TOKEN"]
    for key, value in config["environment_defaults"].items():
        environment.setdefault(key, value)
    if config["use_hf_secret"]:
        token = environment.get(
            "TRUSS_SECRET_hf_access_token", config["hf_secret_default"]
        )
        secret_path = Path("/secrets/hf_access_token")
        if secret_path.is_file():
            token = secret_path.read_text()
        if token:
            environment["BEI_HF_TOKEN"] = token
        elif not (environment.get("BEI_HF_TOKEN") or environment.get("HF_TOKEN")):
            raise ValueError(
                "The declared hf_access_token secret is missing at runtime."
            )
    argv = config["argv"]
    os.execvpe(argv[0], argv, environment)


if __name__ == "__main__":
    main()
