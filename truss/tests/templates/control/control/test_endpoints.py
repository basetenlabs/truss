from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, WebSocket
from httpx import AsyncClient, Response
from httpx_ws import AsyncWebSocketSession
from httpx_ws import _exceptions as httpx_ws_exceptions
from wsproto.events import BytesMessage, TextMessage

from truss.templates.control.control.endpoints import proxy_ws


@pytest.fixture
def app():
    app = FastAPI()
    app.state.proxy_client = AsyncClient(base_url="http://localhost:8080")
    app.state.logger = MagicMock()
    return app


@pytest.fixture
def client_ws(app):
    ws = WebSocket({"type": "websocket", "app": app}, None, None)  # type: ignore
    ws.receive = AsyncMock()
    ws.send_text = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_proxy_ws_bidirectional_messaging(client_ws):
    """Test that both directions of communication work and clean up properly"""
    client_ws.receive.side_effect = [
        {"type": "websocket.receive", "text": "msg1"},
        {"type": "websocket.receive", "text": "msg2"},
        {"type": "websocket.disconnect"},
    ]

    mock_server_ws = AsyncMock(spec=AsyncWebSocketSession)
    mock_server_ws.receive.side_effect = [
        TextMessage(data="response1"),
        TextMessage(data="response2"),
        None,  # server closing connection
    ]
    mock_server_ws.__aenter__.return_value = mock_server_ws
    mock_server_ws.__aexit__.return_value = None

    with patch(
        "truss.templates.control.control.endpoints.aconnect_ws",
        return_value=mock_server_ws,
    ):
        await proxy_ws(client_ws)

    assert mock_server_ws.send_text.call_count == 2
    assert mock_server_ws.send_text.call_args_list == [(("msg1",),), (("msg2",),)]
    assert client_ws.send_text.call_count == 2
    assert client_ws.send_text.call_args_list == [(("response1",),), (("response2",),)]
    client_ws.close.assert_called_once()


@pytest.mark.asyncio
async def test_proxy_ws_binary_message(client_ws):
    test_bytes = b"binary data"
    client_ws.receive.side_effect = [
        {"type": "websocket.receive", "bytes": test_bytes},
        {"type": "websocket.disconnect"},
    ]

    mock_server_ws = AsyncMock(spec=AsyncWebSocketSession)
    mock_server_ws.receive.side_effect = [
        BytesMessage(data=b"binary response"),
        None,  # server closing connection
    ]
    mock_server_ws.__aenter__.return_value = mock_server_ws
    mock_server_ws.__aexit__.return_value = None

    with patch(
        "truss.templates.control.control.endpoints.aconnect_ws",
        return_value=mock_server_ws,
    ):
        await proxy_ws(client_ws)

    # Verify interactions
    client_ws.accept.assert_called_once()
    mock_server_ws.send_bytes.assert_called_once_with(test_bytes)
    client_ws.send_bytes.assert_called_once_with(b"binary response")
    client_ws.close.assert_called_once()


@pytest.mark.asyncio
async def test_proxy_ws_closes_client_upon_server_connection_error(client_ws):
    with patch(
        "truss.templates.control.control.endpoints.aconnect_ws",
        side_effect=httpx_ws_exceptions.WebSocketUpgradeError(
            Response(status_code=503)
        ),
    ):
        await proxy_ws(client_ws)

    client_ws.close.assert_called_once()
    assert client_ws.app.state.logger.warning.called
