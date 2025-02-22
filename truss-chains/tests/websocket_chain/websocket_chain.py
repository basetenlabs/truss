import fastapi

import truss_chains as chains


class Dependency(chains.ChainletBase):
    async def run_remote(self, name: str) -> str:
        msg = f"Hello from dependency, {name}."
        print(msg)
        return msg


@chains.mark_entrypoint  # ("My Chain Name")
class Head(chains.ChainletBase):
    def __init__(self, dependency=chains.depends(Dependency)):
        self._dependency = dependency

    async def run_remote(self, websocket: chains.WebSocketProtocol) -> None:
        try:
            while True:
                text = await websocket.receive_text()
                if text == "dep":
                    result = await self._dependency.run_remote("Head")
                else:
                    result = f"You said: {text}."
                await websocket.send_text(result)
        except fastapi.WebSocketDisconnect:
            print("Disconnected.")
