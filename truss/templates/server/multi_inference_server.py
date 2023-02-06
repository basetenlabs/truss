import os

import yaml
from common.logging import setup_logging
from common.multi_truss_server import MultiTrussServer  # noqa: E402

CONFIG_FILE = "config.yaml"

setup_logging()


class ConfiguredMultiTrussServer:
    _config: dict
    _port: int

    def __init__(self, config_path: str, port: int):
        self._port = port
        with open(config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)

    def start(self):
        server = MultiTrussServer(
            http_port=self._port, truss_configs_by_name={"bola_test": self._config}
        )
        server.start_model()


if __name__ == "__main__":
    env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    ConfiguredMultiTrussServer(CONFIG_FILE, env_port).start()
