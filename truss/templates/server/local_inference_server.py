import os

import yaml
from common.truss_server import TrussServer  # noqa: E402

CONFIG_FILE = "config.yaml"

env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
with open(CONFIG_FILE, encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

server = TrussServer(http_port=env_port, config=config)
app = server.create_application()
