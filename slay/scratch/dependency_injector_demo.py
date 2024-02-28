from typing import reveal_type

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


class ApiClient:
    def __init__(self, api_key: str, timeout: float):
        pass

    def work(self, x: int) -> float:
        return x * 0.1


class Service:
    def __init__(self, api_client: ApiClient) -> None:
        self._client = api_client

    def serve(self, x: int) -> float:
        return x * 0.1


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    api_client = providers.Singleton(
        ApiClient,
        api_key=config.api_key,
        timeout=config.timeout,
    )

    service = providers.Factory(
        Service,
        api_client=api_client,
    )


# reveal_type(Provide[Container.service])
# reveal_type(Provide[Service])


@inject
# def main(service: Service = Provide[Container.service]) -> None:
def main(service: Service = Provide[Service]) -> None:
    print("hi")
    print(service.serve(123))


container = Container()
container.wire(modules=[__name__])
main()
# container.config.api_key.from_env("API_KEY", required=True)
# container.config.timeout.from_env("TIMEOUT", as_=int, default=5)
#
