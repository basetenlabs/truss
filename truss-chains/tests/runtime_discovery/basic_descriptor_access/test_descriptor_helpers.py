"""Smoke tests for the descriptor helpers using ``run_local``-style
construction of a ``DeployedServiceDescriptor``. The chain in ``chain.py``
shows how a chainlet would consume these helpers in production; here we
construct a descriptor directly and assert the helpers' return values."""

import pathlib
import sys

import truss_chains as chains

# Make ``chain.py`` importable from this test file.
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def _make_descriptor(with_internal_url: bool):
    kwargs = {
        "name": "Echo",
        "display_name": "Echo",
        "options": chains.RPCOptions(),
        "predict_url": "https://chain-abc.api.baseten.co/.../echo/run_remote",
    }
    if with_internal_url:
        kwargs["internal_url"] = chains.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://wp.api.baseten.co/.../echo/run_remote",
            hostname="chain-abc.api.baseten.co",
        )
    return chains.DeployedServiceDescriptor(**kwargs)


def test_target_url_with_internal_prefers_internal():
    desc = _make_descriptor(with_internal_url=True)
    assert desc.target_url == "https://wp.api.baseten.co/.../echo/run_remote"


def test_target_url_falls_back_to_predict():
    desc = _make_descriptor(with_internal_url=False)
    assert desc.target_url == "https://chain-abc.api.baseten.co/.../echo/run_remote"


def test_ws_url_scheme_swap():
    desc = _make_descriptor(with_internal_url=False)
    assert desc.ws_url.startswith("wss://")
    assert desc.internal_ws_url is None


def test_internal_ws_url_scheme_swap():
    desc = _make_descriptor(with_internal_url=True)
    assert desc.internal_ws_url.startswith("wss://")


def test_auth_headers_no_internal():
    desc = _make_descriptor(with_internal_url=False)
    assert desc.with_auth_headers("k") == {"Authorization": "Api-Key k"}


def test_auth_headers_with_internal_includes_host():
    desc = _make_descriptor(with_internal_url=True)
    assert desc.with_auth_headers("k") == {
        "Authorization": "Api-Key k",
        "Host": "chain-abc.api.baseten.co",
    }


def test_chain_runs_locally_with_deployed_echo():
    """run_local with a ``chainlet_to_service`` override that points Echo at
    a fake deployed URL — exercises the descriptor helpers via the typed path
    (``context.get_service_descriptor``).

    Caller still calls ``self._echo.run_remote(...)`` over HTTP via the stub
    rather than running Echo in-process, so this scenario is not what you'd
    use in normal local debugging — but it's what makes the descriptor
    helpers observable from a run_local test."""
    from chain import Caller

    fake_descriptor = chains.DeployedServiceDescriptor(
        name="Echo",
        display_name="Echo",
        options=chains.RPCOptions(),
        predict_url="https://chain-abc.api.baseten.co/.../echo/run_remote",
        internal_url=chains.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://wp.api.baseten.co/.../echo/run_remote",
            hostname="chain-abc.api.baseten.co",
        ),
    )

    with chains.run_local(
        secrets={chains.public_types.CHAIN_API_KEY_SECRET_NAME: "test-key"},
        chainlet_to_service={"Echo": fake_descriptor},
    ):
        caller = Caller()

    # We don't actually call run_remote (that would fire HTTP at the fake URL).
    # Inspect the descriptor through the context directly.
    desc = caller._context.get_service_descriptor("Echo")
    assert desc.target_url == "https://wp.api.baseten.co/.../echo/run_remote"
    assert desc.ws_url == "wss://chain-abc.api.baseten.co/.../echo/run_remote"
    assert desc.internal_ws_url == "wss://wp.api.baseten.co/.../echo/run_remote"
    assert desc.with_auth_headers("test-key") == {
        "Authorization": "Api-Key test-key",
        "Host": "chain-abc.api.baseten.co",
    }
