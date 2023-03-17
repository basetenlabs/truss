from truss.server.inference_server import ConfiguredTrussServer

ConfiguredTrussServer("config.yaml", 8080).start()
