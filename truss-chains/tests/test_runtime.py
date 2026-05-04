"""Tests for ``truss_chains.runtime``: the public sibling-discovery API.

Covers the positive paths (descriptor access, URL helpers, list_services) and
the adversarial paths (missing config file, unknown sibling name, empty
mapping).

The fixture ``dynamic_config_mount_dir`` from ``conftest.py`` / shared test
infra monkeypatches ``DYNAMIC_CONFIG_MOUNT_DIR`` to ``tmp_path`` so a test can
write fake JSON to ``tmp_path / "dynamic_chainlet_config"`` and exercise the
runtime exactly as it would behave inside a real chainlet pod.
"""

import json
import pathlib
import sys

import pytest

import truss_chains as chains
from truss_chains import private_types, public_types, runtime

# Make the plain Truss model fixture importable.
_PLAIN_TRUSS_MODEL_DIR = (
    pathlib.Path(__file__).parent / "runtime_discovery" / "plain_truss" / "model"
)
sys.path.insert(0, str(_PLAIN_TRUSS_MODEL_DIR))

# ---- Test fixtures -----------------------------------------------------------

PREDICT_URL_ONLY = {
    "Whisper": {
        "predict_url": "https://chain-abc123.api.baseten.co/deployment/dep/chainlet/cl/run_remote"
    }
}

INTERNAL_AND_PREDICT_URL = {
    "Whisper": {
        "predict_url": "https://chain-abc123.api.baseten.co/deployment/dep/chainlet/cl/run_remote",
        "internal_url": {
            "gateway_run_remote_url": "https://aws-us-west-2-ai7.api.baseten.co/deployment/dep/chainlet/cl/run_remote",
            "hostname": "chain-abc123.api.baseten.co",
        },
    }
}

INTERNAL_URL_ONLY = {
    "Whisper": {
        "internal_url": {
            "gateway_run_remote_url": "https://aws-us-west-2-ai7.api.baseten.co/deployment/dep/chainlet/cl/run_remote",
            "hostname": "chain-abc123.api.baseten.co",
        }
    }
}

MULTIPLE_SIBLINGS = {
    "Whisper": {
        "predict_url": "https://chain-abc.api.baseten.co/.../whisper/run_remote"
    },
    "Diarizer": {
        "predict_url": "https://chain-abc.api.baseten.co/.../diarizer/run_remote",
        "internal_url": {
            "gateway_run_remote_url": "https://wp.api.baseten.co/.../diarizer/run_remote",
            "hostname": "chain-abc.api.baseten.co",
        },
    },
}


@pytest.fixture
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


def _write_config(tmp_path, payload):
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(payload))


# ---- get_service: positive paths --------------------------------------------


def test_get_service_predict_url_only(tmp_path, dynamic_config_mount_dir):
    _write_config(tmp_path, PREDICT_URL_ONLY)
    desc = runtime.get_service("Whisper")
    assert desc.predict_url == PREDICT_URL_ONLY["Whisper"]["predict_url"]
    assert desc.internal_url is None
    assert desc.name == "Whisper"
    assert desc.display_name == "Whisper"


def test_get_service_carries_both_urls_when_present(tmp_path, dynamic_config_mount_dir):
    """The public runtime API faithfully reflects the dynamic config — when the
    config has both predict_url and internal_url, the descriptor carries both.
    (The typed-chain ``populate_chainlet_service_predict_urls`` path retains
    the historical mutually-exclusive behavior; this is tested separately in
    ``test_utils.py``.)"""
    _write_config(tmp_path, INTERNAL_AND_PREDICT_URL)
    desc = runtime.get_service("Whisper")
    assert desc.predict_url == INTERNAL_AND_PREDICT_URL["Whisper"]["predict_url"]
    assert desc.internal_url is not None
    assert (
        desc.internal_url.gateway_run_remote_url
        == INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["gateway_run_remote_url"]
    )
    assert (
        desc.internal_url.hostname
        == INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["hostname"]
    )


def test_get_service_internal_url_only(tmp_path, dynamic_config_mount_dir):
    _write_config(tmp_path, INTERNAL_URL_ONLY)
    desc = runtime.get_service("Whisper")
    assert desc.predict_url is None
    assert desc.internal_url is not None


# ---- get_service: adversarial paths -----------------------------------------


def test_get_service_missing_name_raises(tmp_path, dynamic_config_mount_dir):
    _write_config(tmp_path, MULTIPLE_SIBLINGS)
    with pytest.raises(
        public_types.MissingDependencyError, match="No sibling chainlet named 'Nope'"
    ):
        runtime.get_service("Nope")


def test_get_service_missing_name_lists_available(tmp_path, dynamic_config_mount_dir):
    """Error message must include the list of available chainlet names so
    users can debug typos quickly."""
    _write_config(tmp_path, MULTIPLE_SIBLINGS)
    with pytest.raises(public_types.MissingDependencyError) as excinfo:
        runtime.get_service("Nope")
    assert "Whisper" in str(excinfo.value)
    assert "Diarizer" in str(excinfo.value)


def test_get_service_no_chain_context_raises(tmp_path, dynamic_config_mount_dir):
    # No file written.
    with pytest.raises(
        public_types.MissingDependencyError, match="not running inside a chain context"
    ):
        runtime.get_service("Whisper")


# ---- list_services ----------------------------------------------------------


def test_list_services_empty_when_no_context(tmp_path, dynamic_config_mount_dir):
    """list_services must NOT raise outside a chain — returns empty mapping."""
    assert runtime.list_services() == {}


def test_list_services_returns_all(tmp_path, dynamic_config_mount_dir):
    _write_config(tmp_path, MULTIPLE_SIBLINGS)
    services = runtime.list_services()
    assert set(services.keys()) == {"Whisper", "Diarizer"}
    assert services["Whisper"].predict_url is not None
    assert services["Diarizer"].internal_url is not None


# ---- DeployedServiceDescriptor helpers --------------------------------------


def test_descriptor_target_url_prefers_internal():
    """target_url mirrors BasetenSession's selection: internal_url wins."""
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        predict_url="https://public.example/predict",
        internal_url=public_types.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://internal.example/predict",
            hostname="public.example",
        ),
    )
    assert desc.target_url == "https://internal.example/predict"


def test_descriptor_target_url_falls_back_to_predict():
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        predict_url="https://public.example/predict",
    )
    assert desc.target_url == "https://public.example/predict"


def test_descriptor_ws_url_https_to_wss():
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        predict_url="https://public.example/predict",
    )
    assert desc.ws_url == "wss://public.example/predict"
    assert desc.internal_ws_url is None


def test_descriptor_ws_url_http_to_ws():
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        predict_url="http://localhost:8080/predict",
    )
    assert desc.ws_url == "ws://localhost:8080/predict"


def test_descriptor_internal_ws_url():
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        internal_url=public_types.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://internal.example/predict",
            hostname="public.example",
        ),
    )
    assert desc.internal_ws_url == "wss://internal.example/predict"
    assert desc.ws_url is None


def test_descriptor_with_auth_headers_predict_only():
    """Without internal_url, only Authorization header is set."""
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        predict_url="https://public.example/predict",
    )
    headers = desc.with_auth_headers("my-key")
    assert headers == {"Authorization": "Api-Key my-key"}


def test_descriptor_with_auth_headers_with_internal_url():
    """With internal_url, both Authorization and Host are set — Host carries
    the chain hostname so cluster-local routing matches the chain identity."""
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        internal_url=public_types.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://internal.example/predict",
            hostname="chain-abc.api.baseten.co",
        ),
    )
    headers = desc.with_auth_headers("my-key")
    assert headers == {
        "Authorization": "Api-Key my-key",
        "Host": "chain-abc.api.baseten.co",
    }


def test_with_auth_headers_matches_BasetenSession():
    """Mechanically verify ``with_auth_headers`` produces the exact dict
    ``BasetenSession.__init__`` would assemble — preventing drift between the
    helper and the framework's internal RPC client."""
    from truss_chains.remote_chainlet import stub

    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        internal_url=public_types.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://internal.example/predict",
            hostname="chain-abc.api.baseten.co",
        ),
    )
    session = stub.BasetenSession(service_descriptor=desc, api_key="my-key")
    assert session._headers == desc.with_auth_headers("my-key")


def test_descriptor_ws_url_unknown_scheme_raises():
    desc = public_types.DeployedServiceDescriptor(
        name="X",
        display_name="X",
        options=public_types.RPCOptions(),
        predict_url="ftp://nope.example/predict",
    )
    with pytest.raises(ValueError, match="Cannot convert to WebSocket scheme"):
        _ = desc.ws_url


def test_descriptor_type_identity_across_imports():
    """One canonical class, regardless of import path."""
    from truss_chains import DeployedServiceDescriptor as A
    from truss_chains.public_types import DeployedServiceDescriptor as B

    assert A is B


# ---- run_local: typed dependency exposes descriptor helpers ------------------


class _Echo(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi")
    )

    async def run_remote(self, text: str) -> str:
        return text


class _Caller(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi")
    )

    def __init__(
        self,
        echo: _Echo = chains.depends(_Echo),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._echo = echo
        self._context = context

    async def run_remote(self, text: str) -> str:
        return await self._echo.run_remote(text)


def test_chain_runs_locally_with_deployed_dep():
    """``run_local`` with a ``chainlet_to_service`` override exposes the
    descriptor helpers via the typed path (``context.get_service_descriptor``)."""
    fake_descriptor = chains.DeployedServiceDescriptor(
        name="_Echo",
        display_name="_Echo",
        options=chains.RPCOptions(),
        predict_url="https://chain-abc.api.baseten.co/.../echo/run_remote",
        internal_url=chains.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://wp.api.baseten.co/.../echo/run_remote",
            hostname="chain-abc.api.baseten.co",
        ),
    )

    with chains.run_local(
        secrets={public_types.CHAIN_API_KEY_SECRET_NAME: "test-key"},
        chainlet_to_service={"_Echo": fake_descriptor},
    ):
        caller = _Caller()

    desc = caller._context.get_service_descriptor("_Echo")
    assert desc.target_url == "https://wp.api.baseten.co/.../echo/run_remote"
    assert desc.ws_url == "wss://chain-abc.api.baseten.co/.../echo/run_remote"
    assert desc.internal_ws_url == "wss://wp.api.baseten.co/.../echo/run_remote"
    assert desc.with_auth_headers("test-key") == {
        "Authorization": "Api-Key test-key",
        "Host": "chain-abc.api.baseten.co",
    }


# ---- Plain Truss bring-your-own-client -------------------------------------


def test_plain_truss_picks_up_siblings(tmp_path, dynamic_config_mount_dir):
    """A non-ChainletBase Truss reads sibling URLs via ``runtime.list_services``
    and ``runtime.get_service`` — the bring-your-own-client pattern."""
    _write_config(
        tmp_path,
        {
            "Diarizer": {
                "predict_url": "https://chain-abc.api.baseten.co/.../diarizer/run_remote",
                "internal_url": {
                    "gateway_run_remote_url": "https://wp.api.baseten.co/.../diarizer/run_remote",
                    "hostname": "chain-abc.api.baseten.co",
                },
            }
        },
    )

    from model import Model  # type: ignore[import-not-found]

    m = Model()
    m.load()
    out = m.predict({})
    assert out["diarizer_url"] == "https://wp.api.baseten.co/.../diarizer/run_remote"
    assert out["auth_headers"] == {
        "Authorization": "Api-Key <from-secrets>",
        "Host": "chain-abc.api.baseten.co",
    }


def test_plain_truss_runs_standalone(tmp_path, dynamic_config_mount_dir):
    """No dynamic config file present — ``list_services`` returns empty,
    Model.load short-circuits, predict returns the standalone shape."""
    from model import Model  # type: ignore[import-not-found]

    m = Model()
    m.load()
    out = m.predict({})
    assert out["diarizer_url"] is None
    assert out["auth_headers"] is None
