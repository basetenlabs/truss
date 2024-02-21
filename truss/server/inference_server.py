import os
from pathlib import Path
from typing import Dict

from truss.server.common.truss_server import TrussServer
from truss.server.shared.logging import setup_logging
from truss.truss_config import TrussConfig

CONFIG_FILE = "config.yaml"

setup_logging()


class ConfiguredTrussServer:
    _config: Dict
    _port: int

    def __init__(self, config_path: str, port: int):
        self._port = port
        self._config = TrussConfig.from_yaml(Path(config_path)).to_dict(verbose=True)

    def start(self):
        server = TrussServer(http_port=self._port, config=self._config)
        server.start()


if __name__ == "__main__":
    env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    ConfiguredTrussServer(CONFIG_FILE, env_port).start()
