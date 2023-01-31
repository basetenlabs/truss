import os

import yaml
from common.util import setup_logging

setup_logging()

from common.truss_server import TrussServer  # noqa: E402
from model_wrapper import ModelWrapper  # noqa: E402

CONFIG_FILE = "config.yaml"


class ConfiguredTrussServer:
    _config: dict
    _port: int

    def __init__(self, config_path: str, port: int):
        self._port = port
        with open(config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)

    def start(self):
        model = ModelWrapper(self._config)
        server = TrussServer(http_port=self._port, model=model)
        server.start_model()


if __name__ == "__main__":
    env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    ConfiguredTrussServer(CONFIG_FILE, env_port).start()
