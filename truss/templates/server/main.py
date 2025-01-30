import os

from truss_server import TrussServer

CONFIG_FILE = "config.yaml"

if __name__ == "__main__":
    http_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    TrussServer(http_port, CONFIG_FILE).start()
