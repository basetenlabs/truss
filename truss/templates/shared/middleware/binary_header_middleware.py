from typing import Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class BinaryHeaderMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, map_input: bool = True, map_output: bool = True):
        super().__init__(app)
        self.map_input = map_input
        self.map_output = map_output

    @staticmethod
    def is_binary(request: Request) -> Tuple[bool, Optional[str]]:
        content_type_header = request.headers.get("content-type")
        return (content_type_header == "application/octet-stream", content_type_header)

    async def dispatch(self, request, call_next):
        is_binary, original_content_type = self.is_binary(request)

        headers = dict(request.scope["headers"])

        if is_binary and self.map_input:
            headers[b"content-type"] = b"application/x-msgpack"

        if is_binary and self.map_output:
            headers[b"accept"] = b"application/x-msgpack"

        request.scope["headers"] = [(k, v) for k, v in headers.items()]

        response = await call_next(request)

        if is_binary:
            response.headers.append("content-type", original_content_type)
        return response
