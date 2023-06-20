import asyncio
import concurrent.futures
import logging
import multiprocessing
import socket
from typing import Any, Dict, List, Optional

import shared.util as utils
import uvicorn
from fastapi import FastAPI
from gunicorn.app.base import BaseApplication

FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s [%(funcName)s():%(lineno)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


class UvicornCustomServer(multiprocessing.Process):
    def __init__(
        self, config: uvicorn.Config, sockets: Optional[List[socket.socket]] = None
    ):
        super().__init__()
        self.sockets = sockets
        self.config = config

    def stop(self):
        self.terminate()

    def run(self):
        server = uvicorn.Server(config=self.config)
        asyncio.run(server.serve(sockets=self.sockets))


class StandaloneApplication(BaseApplication):
    def __init__(self, application: FastAPI, options: Dict[str, Any] = None):
        self.options = options or {}
        self.application = application
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1


def start_uvicorn_server(application: FastAPI, host: str = "*", port: int = 8080):

    # StandaloneApplication(
    #     application,
    #     {
    #         "bind": "%s:%s" % (host, port),
    #         "workers": 1,
    #         "worker_class": "uvicorn.workers.UvicornWorker",
    #     },
    # ).run()
    # cfg = uvicorn.Config(
    #     application,
    #     host=host,
    #     port=port,
    #     workers=1,
    # )

    # max_asyncio_workers = min(32, multiprocessing.cpu_count() + 4)
    # cfg.setup_event_loop()
    # asyncio.get_event_loop().set_default_executor(
    #     concurrent.futures.ThreadPoolExecutor(max_workers=max_asyncio_workers)
    # )
    # uvicorn.run(cfg)

    cfg = uvicorn.Config(
        application,
        host=host,
        port=port,
        workers=1,
    )

    max_asyncio_workers = min(32, utils.cpu_count() + 4)
    logging.info(f"Setting max asyncio worker threads as {max_asyncio_workers}")
    # Call this so uvloop gets used
    cfg.setup_event_loop()
    asyncio.get_event_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_asyncio_workers)
    )

    async def serve():
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serversocket.bind((cfg.host, cfg.port))
        serversocket.listen(5)

        logging.info(f"starting uvicorn with {cfg.workers} workers")
        for _ in range(cfg.workers):
            server = UvicornCustomServer(config=cfg, sockets=[serversocket])
            server.start()

    async def servers_task():
        servers = [serve()]
        await asyncio.gather(*servers)

    asyncio.run(servers_task())
