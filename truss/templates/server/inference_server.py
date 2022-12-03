import os

import yaml
from common.truss_server import TrussServer
from model_wrapper import ModelWrapper

CONFIG_FILE = "config.yaml"


class ConfiguredTrussServer:
    _config: dict
    _port: int

    def __init__(self, config_path: str, port: int):
        self._port = port
        with open(config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)

    def start(self):
        server = TrussServer(workers=1, http_port=self._port)
        model = ModelWrapper(self._config)
        server.start([model])


if __name__ == "__main__":
    env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    ConfiguredTrussServer(CONFIG_FILE, env_port).start()
