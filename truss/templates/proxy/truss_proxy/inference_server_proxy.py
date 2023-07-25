import re
from typing import Callable, List, Optional, Tuple, Union

from proxy.common.types import RePattern
from proxy.http import Url
from proxy.http.exception import HttpRequestRejected
from proxy.http.parser import HttpParser
from proxy.http.server import ReverseProxyBasePlugin


class InferenceServerReverseProxyPlugin(ReverseProxyBasePlugin):
    inference_server_path = b"http://localhost:8090/"
    control_server_path = b"http://localhost:8091/"
    server_down_callback: Optional[Callable] = None

    def routes(self) -> List[Union[str, Tuple[str, List[bytes]]]]:
        static_control_routes = [
            (
                rf"/control/{method}$",
                [
                    InferenceServerReverseProxyPlugin.control_server_path
                    + f"control/{method}".encode()
                ],
            )
            for method in [
                "proxy",
                "truss_hash",
                "restart_inference_server",
                "has_partially_applied_patch",
                "stop_inference_server",
            ]
        ]
        return static_control_routes + [
            r"/v1/(.+)$",
        ]

    def handle_route(self, request: HttpParser, pattern: RePattern) -> Url:
        """For our example dynamic route, we want to simply convert
        any incoming request to "/get/1" into "/get?id=1" when serving from upstream.
        """
        choice: Url = Url.from_bytes(
            InferenceServerReverseProxyPlugin.inference_server_path
        )
        if (
            InferenceServerReverseProxyPlugin.server_down_callback is not None
            and InferenceServerReverseProxyPlugin.server_down_callback()
        ):
            raise HttpRequestRejected(
                503, b"Model Load Failed", body=b'{"error": "Model Load Failed"}'
            )
        result = re.search(pattern, request.path.decode())
        print(result)
        choice.remainder += f"v1/{result.groups()[0]}".encode()
        return choice
