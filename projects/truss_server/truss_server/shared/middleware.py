from msgpack_asgi import MessagePackMiddleware
from shared.serialization import truss_msgpack_deserialize, truss_msgpack_serialize
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

OCTET_STREAN_CONTENT_TYPE = "application/octet-stream"
MSGPACK_CONTENT_TYPE = b"application/x-msgpack"


class BinaryHeaderMiddleware(BaseHTTPMiddleware):
    """ASGIMiddleware to map binary requests to msgpack.

    Note:
    Using BaseHTTPMiddleware will prevent changes to
    [contextlib.ContextVars](https://docs.python.org/3/library/contextvars.html#contextvars.ContextVar)
    from propagating upwards. That is, if you set a value for a ContextVar in your endpoint and try to
    read it from a middleware you will find that the value is not the same value you set in your endpoint (see
    [this test](https://github.com/encode/starlette/blob/master/tests/middleware/test_base.py#L192-L223)
    for an example of this behavior).
    """

    def __init__(
        self,
        app,
        map_input: bool = True,
        map_output: bool = True,
    ):
        super().__init__(app)
        self.map_input = map_input
        self.map_output = map_output

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        is_binary = OCTET_STREAN_CONTENT_TYPE in (
            request.headers.get("content-type") or []
        )

        headers = dict(request.scope["headers"])

        if is_binary and self.map_input:
            headers[b"content-type"] = MSGPACK_CONTENT_TYPE

        if is_binary and self.map_output:
            headers[b"accept"] = MSGPACK_CONTENT_TYPE

        request.scope["headers"] = [(k, v) for k, v in headers.items()]

        response = await call_next(request)

        if is_binary:
            response.headers.append("content-type", OCTET_STREAN_CONTENT_TYPE)
        return response


class TrussMsgpackMiddleware(MessagePackMiddleware):
    """MessagePackMiddleware with custom truss packb and unpackb

    An ASGI application wrapped around MessagePackMiddleware will perform automatic content negotiation based on the
    client's capabilities. More precisely:
    1. If the client sends MessagePack-encoded data with the application/x-msgpack content type,
       msgpack-asgi will automatically re-encode the body to JSON and re-write the request Content-Type
       to application/json for your application to consume. (Note: this means applications will not be
       able to distinguish between MessagePack and JSON client requests.)
    2. If the client sent the Accept: application/x-msgpack header, msgpack-asgi will automatically re-encode
    any JSON response data to MessagePack for the client to consume.
    (In other cases, msgpack-asgi won't intervene at all.)
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(
            app,
            packb=truss_msgpack_serialize,
            unpackb=truss_msgpack_deserialize,
        )
