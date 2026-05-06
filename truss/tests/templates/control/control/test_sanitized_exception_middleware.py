from unittest import mock

import pytest

from truss.tests.templates.control.control.conftest import setup_control_imports

setup_control_imports()

from truss.templates.control.control.application import (  # noqa: E402
    SanitizedExceptionMiddleware,
)


@pytest.fixture
def middleware():
    return SanitizedExceptionMiddleware(app=mock.Mock(), num_frames=2)


def _request(method: str = "GET", path: str = "/v1/models/model/loaded"):
    req = mock.Mock()
    req.method = method
    req.url.path = path
    return req


def _raise_chain():
    """Raise a 3-level cause chain: ConnectError -> ProtocolError -> RuntimeError."""
    try:
        try:
            raise ConnectionError("dns failed")
        except ConnectionError as inner:
            raise OSError("protocol error") from inner
    except OSError as middle:
        raise RuntimeError("outer") from middle


def test_format_simple_exception_no_chain(middleware):
    try:
        raise ValueError("bad input")
    except ValueError as exc:
        out = middleware._format_error(_request("POST", "/control/patch"), exc)

    lines = out.splitlines()
    assert lines[0] == "POST /control/patch: ValueError: bad input"
    # No "Caused by:" lines for a single-exception chain.
    assert not any(line.startswith("Caused by:") for line in lines)
    # Frame lines have no caret/squiggle markers (PEP 657).
    assert "^" not in out


def test_format_walks_cause_chain(middleware):
    try:
        _raise_chain()
    except RuntimeError as exc:
        out = middleware._format_error(_request(), exc)

    assert out.startswith("GET /v1/models/model/loaded: RuntimeError: outer")
    # All three causes appear with their type and message.
    assert "Caused by: OSError: protocol error" in out
    assert "Caused by: ConnectionError: dns failed" in out
    # Headlines are listed outermost-to-innermost.
    assert out.index("OSError") < out.index("ConnectionError")
    # No carets.
    assert "^" not in out


def test_frame_count_bounded_per_exception(middleware):
    try:
        _raise_chain()
    except RuntimeError as exc:
        out = middleware._format_error(_request(), exc)

    frame_lines = [
        line for line in out.splitlines() if line.lstrip().startswith("File ")
    ]
    # 3 exceptions in chain * num_frames(=2) = at most 6 frame lines. The
    # innermost frames may have fewer (single-statement raises), so allow <=.
    assert 1 <= len(frame_lines) <= 6


def test_cycle_detection_terminates(middleware):
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # cycle
    a.__traceback__ = None
    b.__traceback__ = None

    out = middleware._format_error(_request(), a)
    # Each appears exactly once despite the cycle.
    assert out.count("RuntimeError: a") == 1
    assert out.count("RuntimeError: b") == 1


def test_self_referential_cause_terminates(middleware):
    e = RuntimeError("self")
    e.__cause__ = e
    out = middleware._format_error(_request(), e)
    assert out.count("RuntimeError: self") == 1


def test_suppress_context_skips_context(middleware):
    # `raise X from None` sets __suppress_context__=True; the implicit context
    # should not be walked.
    try:
        try:
            raise ValueError("hidden")
        except ValueError:
            raise RuntimeError("outer") from None
    except RuntimeError as exc:
        out = middleware._format_error(_request(), exc)

    assert "RuntimeError: outer" in out
    assert "ValueError: hidden" not in out
    assert "Caused by:" not in out


def test_num_frames_zero_disables_frames(middleware):
    middleware.num_frames = 0
    try:
        _raise_chain()
    except RuntimeError as exc:
        out = middleware._format_error(_request(), exc)

    frame_lines = [
        line for line in out.splitlines() if line.lstrip().startswith("File ")
    ]
    assert frame_lines == []
    # Headlines for all exceptions still present.
    assert "RuntimeError: outer" in out
    assert "OSError: protocol error" in out
    assert "ConnectionError: dns failed" in out
