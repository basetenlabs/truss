import os
from pathlib import Path

import yaml
from common.logging import setup_logging
from common.multi_truss_server import MultiTrussServer  # noqa: E402

CONFIG_FILE = "config.yaml"

DEFAULT_MODEL_REPO_DIR = "/models/"


setup_logging()


class ConfiguredMultiTrussServer:
    _config: dict
    _port: int
    _model_dir: Path

    def __init__(self, config_path: str, port: int, model_dir: str):
        self._port = port
        self._model_dir = Path(model_dir)
        with open(config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)

    def start(self):
        server = MultiTrussServer(
            http_port=self._port,
            multi_truss_config=self._config,
            model_truss_dir=self._model_dir,
        )
        server.start_model()


if __name__ == "__main__":
    env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    ConfiguredMultiTrussServer(CONFIG_FILE, env_port, DEFAULT_MODEL_REPO_DIR).start()
