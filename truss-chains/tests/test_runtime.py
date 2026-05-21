"""Tests for ``truss_chains.runtime`` (sibling URL discovery from dynamic config)."""

import json
import pathlib
import sys

import pytest

import truss_chains as chains
from truss_chains import private_types, public_types, runtime

_PLAIN_TRUSS_MODEL_DIR = (
    pathlib.Path(__file__).parent / "runtime_discovery" / "plain_truss" / "model"
)

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
    # ``runtime.load_dynamic_chainlet_config`` is ``lru_cache``-decorated;
    # invalidate before and after each test so monkeypatched paths/contents
    # take effect, and so the cache doesn't leak across tests.
    runtime.load_dynamic_chainlet_config.cache_clear()
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield
    runtime.load_dynamic_chainlet_config.cache_clear()


@pytest.fixture
def plain_truss_model(monkeypatch: pytest.MonkeyPatch):
    """Import ``model.Model`` from the plain Truss fixture with isolated ``sys.path``."""
    monkeypatch.syspath_prepend(str(_PLAIN_TRUSS_MODEL_DIR))
    monkeypatch.delitem(sys.modules, "model", raising=False)
    from model import Model  # type: ignore[import-not-found]

    yield Model
    sys.modules.pop("model", None)


def _write_config(tmp_path, payload):
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(payload))


# ---- get_service_urls: positive paths ----------------------------------------


def test_get_service_urls_predict_url_only(tmp_path, dynamic_config_mount_dir):
    _write_config(tmp_path, PREDICT_URL_ONLY)
    urls = runtime.get_service_urls("Whisper")
    assert urls.predict_url == PREDICT_URL_ONLY["Whisper"]["predict_url"]
    assert urls.internal_url is None


def test_get_service_urls_carries_both_urls_when_present(
    tmp_path, dynamic_config_mount_dir
):
    """Result includes ``predict_url`` and ``internal_url`` when config has both."""
    _write_config(tmp_path, INTERNAL_AND_PREDICT_URL)
    urls = runtime.get_service_urls("Whisper")
    assert urls.predict_url == INTERNAL_AND_PREDICT_URL["Whisper"]["predict_url"]
    assert urls.internal_url is not None
    assert (
        urls.internal_url.gateway_run_remote_url
        == INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["gateway_run_remote_url"]
    )
    assert (
        urls.internal_url.hostname
        == INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["hostname"]
    )


def test_get_service_urls_internal_url_only(tmp_path, dynamic_config_mount_dir):
    _write_config(tmp_path, INTERNAL_URL_ONLY)
    urls = runtime.get_service_urls("Whisper")
    assert urls.predict_url is None
    assert urls.internal_url is not None


# ---- get_service_urls: adversarial paths ------------------------------------


def test_get_service_urls_missing_name_raises(tmp_path, dynamic_config_mount_dir):
    _write_config(tmp_path, MULTIPLE_SIBLINGS)
    with pytest.raises(
        public_types.MissingDependencyError, match="No sibling chainlet named 'Nope'"
    ):
        runtime.get_service_urls("Nope")


def test_get_service_urls_missing_name_lists_available(
    tmp_path, dynamic_config_mount_dir
):
    """``MissingDependencyError`` lists known sibling names."""
    _write_config(tmp_path, MULTIPLE_SIBLINGS)
    with pytest.raises(public_types.MissingDependencyError) as excinfo:
        runtime.get_service_urls("Nope")
    assert "Whisper" in str(excinfo.value)
    assert "Diarizer" in str(excinfo.value)


def test_get_service_urls_no_chain_context_raises(tmp_path, dynamic_config_mount_dir):
    # No file written.
    with pytest.raises(
        public_types.MissingDependencyError, match="Cannot override Chainlet configs"
    ):
        runtime.get_service_urls("Whisper")


def test_get_service_urls_invalid_input_type_raises():
    """Invalid ``target`` type raises ``TypeError``."""
    with pytest.raises(TypeError, match="chainlet class or its display-name"):
        runtime.get_service_urls(42)  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", ["", [], "some string"])
def test_load_dynamic_chainlet_config_treats_non_dict_as_empty(
    tmp_path, dynamic_config_mount_dir, payload
):
    """JSON roots that aren't objects are treated like an empty mapping."""
    _write_config(tmp_path, payload)
    with pytest.raises(
        public_types.MissingDependencyError, match="No sibling chainlet named"
    ):
        runtime.get_service_urls("Whisper")


# ---- get_service_urls: class-keyed lookup -----------------------------------


def test_get_service_urls_by_class_uses_class_name(tmp_path, dynamic_config_mount_dir):
    """Chainlet class target resolves keys by ``display_name`` (defaults to ``__name__``)."""
    _write_config(tmp_path, PREDICT_URL_ONLY)

    class Whisper(chains.ChainletBase):
        async def run_remote(self, text: str) -> str:
            return text

    urls = runtime.get_service_urls(Whisper)
    assert urls.predict_url == PREDICT_URL_ONLY["Whisper"]["predict_url"]


def test_get_service_urls_by_class_honors_display_name_override(
    tmp_path, dynamic_config_mount_dir
):
    """``RemoteConfig.name`` becomes ``display_name`` for class-keyed lookup."""
    _write_config(
        tmp_path,
        {
            "diarizer-v2": {
                "predict_url": "https://chain-x.api.baseten.co/.../diarizer/run_remote"
            }
        },
    )

    class Diarizer(chains.ChainletBase):
        remote_config = chains.RemoteConfig(name="diarizer-v2")

        async def run_remote(self, text: str) -> str:
            return text

    urls = runtime.get_service_urls(Diarizer)
    assert urls.predict_url == "https://chain-x.api.baseten.co/.../diarizer/run_remote"


# ---- get_baseten_chain_api_key ----------------------------------------------


@pytest.fixture
def clear_chain_api_key_cache():
    """Clear ``get_baseten_chain_api_key`` cache around the test."""
    runtime.get_baseten_chain_api_key.cache_clear()
    yield
    runtime.get_baseten_chain_api_key.cache_clear()


def test_get_baseten_chain_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch, clear_chain_api_key_cache
):
    """API key from ``TRUSS_SECRET_*`` env matches secrets resolver behavior."""
    monkeypatch.setenv(
        f"TRUSS_SECRET_{public_types.CHAIN_API_KEY_SECRET_NAME}", "secret123"
    )
    assert runtime.get_baseten_chain_api_key() == "secret123"


def test_get_baseten_chain_api_key_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path, clear_chain_api_key_cache
):
    """Missing secret raises ``MissingDependencyError``."""
    monkeypatch.delenv(
        f"TRUSS_SECRET_{public_types.CHAIN_API_KEY_SECRET_NAME}", raising=False
    )
    # Point secrets mount at an empty tmp dir so the file lookup also fails.
    monkeypatch.setattr(
        "truss.templates.shared.secrets_resolver.SecretsResolver.SECRETS_MOUNT_DIR",
        str(tmp_path),
    )
    with pytest.raises(
        public_types.MissingDependencyError,
        match=f"No '{public_types.CHAIN_API_KEY_SECRET_NAME}'",
    ):
        runtime.get_baseten_chain_api_key()


# ---- ServiceHandle ----------------------------------------------------------


@pytest.fixture
def _stub_api_key(monkeypatch: pytest.MonkeyPatch, clear_chain_api_key_cache):
    """Fixed chain API key in env for ``http_call_args`` assertions."""
    monkeypatch.setenv(
        f"TRUSS_SECRET_{public_types.CHAIN_API_KEY_SECRET_NAME}", "test-key"
    )
    yield "test-key"


def test_service_handle_exposes_raw_urls(tmp_path, dynamic_config_mount_dir):
    """``.urls`` exposes the parsed ``ServiceDescriptorUrls``."""
    _write_config(tmp_path, INTERNAL_AND_PREDICT_URL)
    handle = runtime.ServiceHandle("Whisper")
    assert handle.urls.predict_url == INTERNAL_AND_PREDICT_URL["Whisper"]["predict_url"]
    assert handle.urls.internal_url is not None
    assert (
        handle.urls.internal_url.hostname
        == INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["hostname"]
    )


def test_service_handle_missing_target_raises(tmp_path, dynamic_config_mount_dir):
    """Unknown sibling name fails in ``ServiceHandle()``."""
    _write_config(tmp_path, MULTIPLE_SIBLINGS)
    with pytest.raises(
        public_types.MissingDependencyError, match="No sibling chainlet named 'Nope'"
    ):
        runtime.ServiceHandle("Nope")


def test_service_handle_constructor_takes_class(tmp_path, dynamic_config_mount_dir):
    """``ServiceHandle`` accepts a chainlet class (distinct name avoids registry clashes)."""
    expected_url = "https://chain-xyz.api.baseten.co/.../whisper/run_remote"
    _write_config(tmp_path, {"WhisperHandle": {"predict_url": expected_url}})

    class WhisperHandle(chains.ChainletBase):
        async def run_remote(self, text: str) -> str:
            return text

    handle = runtime.ServiceHandle(WhisperHandle)
    assert handle.urls.predict_url == expected_url


def test_http_call_args_default_uses_predict_url(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """Default uses ``predict_url`` and ``Authorization`` only."""
    _write_config(tmp_path, INTERNAL_AND_PREDICT_URL)
    call = runtime.ServiceHandle("Whisper").http_call_args()
    assert call.url == INTERNAL_AND_PREDICT_URL["Whisper"]["predict_url"]
    assert call.headers == {"Authorization": "Api-Key test-key"}


def test_http_call_args_prefer_internal_uses_workload_plane_url(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``prefer_internal`` uses workload gateway URL plus ``Host`` header."""
    _write_config(tmp_path, INTERNAL_AND_PREDICT_URL)
    call = runtime.ServiceHandle("Whisper").http_call_args(prefer_internal=True)
    assert (
        call.url
        == INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["gateway_run_remote_url"]
    )
    assert call.headers == {
        "Authorization": "Api-Key test-key",
        "Host": INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["hostname"],
    }


def test_http_call_args_predict_only_ignores_prefer_internal(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """No ``internal_url`` means ``prefer_internal`` still uses ``predict_url``."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").http_call_args(prefer_internal=True)
    assert call.url == PREDICT_URL_ONLY["Whisper"]["predict_url"]
    assert call.headers == {"Authorization": "Api-Key test-key"}


def test_http_call_args_internal_only_falls_through_silently(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``predict_url`` missing falls back to internal gateway URL + ``Host``."""
    _write_config(tmp_path, INTERNAL_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").http_call_args()
    assert (
        call.url
        == INTERNAL_URL_ONLY["Whisper"]["internal_url"]["gateway_run_remote_url"]
    )
    assert call.headers == {
        "Authorization": "Api-Key test-key",
        "Host": INTERNAL_URL_ONLY["Whisper"]["internal_url"]["hostname"],
    }


def test_http_call_args_api_key_override(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``api_key=`` overrides the resolved secret."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").http_call_args(api_key="override-key")
    assert call.headers == {"Authorization": "Api-Key override-key"}


def test_http_call_args_is_destructurable(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """Return value unpacks as ``url, headers``."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    url, headers = runtime.ServiceHandle("Whisper").http_call_args()
    assert url == PREDICT_URL_ONLY["Whisper"]["predict_url"]
    assert "Authorization" in headers


def test_http_call_args_sync_path_rewrites_url(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``sync_path`` rewrites the URL from ``/run_remote`` to ``/sync/<path>``,
    the alternative routing for TC siblings that expose their predict
    endpoint via the platform's path-passthrough."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").http_call_args(
        sync_path="v1/models/model:predict"
    )
    expected = PREDICT_URL_ONLY["Whisper"]["predict_url"].removesuffix("/run_remote")
    assert call.url == f"{expected}/sync/v1/models/model:predict"
    # Auth header unaffected; no Host (predict_url path).
    assert call.headers == {"Authorization": "Api-Key test-key"}


def test_http_call_args_sync_path_with_prefer_internal(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``sync_path`` combined with ``prefer_internal=True`` rewrites the
    workload-plane URL the same way, and the ``Host`` header is preserved."""
    _write_config(tmp_path, INTERNAL_AND_PREDICT_URL)
    call = runtime.ServiceHandle("Whisper").http_call_args(
        prefer_internal=True, sync_path="predict"
    )
    base = INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"][
        "gateway_run_remote_url"
    ].removesuffix("/run_remote")
    assert call.url == f"{base}/sync/predict"
    assert (
        call.headers["Host"]
        == (INTERNAL_AND_PREDICT_URL["Whisper"]["internal_url"]["hostname"])
    )


def test_http_call_args_sync_path_leading_slash_normalized(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``sync_path`` accepts both ``"foo"`` and ``"/foo"``; the leading slash
    is stripped to produce a single ``/sync/`` prefix."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").http_call_args(sync_path="/predict")
    expected = PREDICT_URL_ONLY["Whisper"]["predict_url"].removesuffix("/run_remote")
    assert call.url == f"{expected}/sync/predict"


# ---- ws_call_args ------------------------------------------------------------


def test_ws_call_args_rewrites_scheme_and_path(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``ws_call_args`` returns a ``wss://`` URL with the trailing ``/run_remote``
    rewritten to ``/websocket``. Headers carry only ``Authorization`` (no
    ``Host`` for the predict_url path)."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").ws_call_args()
    expected_base = (
        PREDICT_URL_ONLY["Whisper"]["predict_url"]
        .removesuffix("/run_remote")
        .replace("https://", "wss://")
    )
    assert call.url == f"{expected_base}/websocket"
    assert call.headers == {"Authorization": "Api-Key test-key"}


def test_ws_call_args_ignores_internal_url(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``ws_call_args`` does NOT honor ``internal_url`` even when present —
    ``websockets.connect`` rejects Host-header overrides (api-gateway returns
    400), so WS BYOC always uses ``predict_url`` (chain-host)."""
    _write_config(tmp_path, INTERNAL_AND_PREDICT_URL)
    call = runtime.ServiceHandle("Whisper").ws_call_args()
    expected_base = (
        INTERNAL_AND_PREDICT_URL["Whisper"]["predict_url"]
        .removesuffix("/run_remote")
        .replace("https://", "wss://")
    )
    assert call.url == f"{expected_base}/websocket"
    # Auth-only — no Host header.
    assert call.headers == {"Authorization": "Api-Key test-key"}


def test_ws_call_args_sync_path_keeps_sync_route(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``sync_path`` overrides the default ``/websocket`` rewrite; scheme is
    still rewritten to ``wss://`` so the caller can use ``websockets.connect``."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").ws_call_args(sync_path="v1/websocket")
    expected_base = (
        PREDICT_URL_ONLY["Whisper"]["predict_url"]
        .removesuffix("/run_remote")
        .replace("https://", "wss://")
    )
    assert call.url == f"{expected_base}/sync/v1/websocket"


def test_ws_call_args_api_key_override(
    tmp_path, dynamic_config_mount_dir, _stub_api_key
):
    """``api_key=`` overrides the resolved secret (parity with http_call_args)."""
    _write_config(tmp_path, PREDICT_URL_ONLY)
    call = runtime.ServiceHandle("Whisper").ws_call_args(api_key="override-key")
    assert call.headers == {"Authorization": "Api-Key override-key"}


# ---- DeployedServiceDescriptor type identity --------------------------------


def test_descriptor_type_identity_across_imports():
    """``DeployedServiceDescriptor`` is the same class via alternate imports."""
    from truss_chains import DeployedServiceDescriptor as A
    from truss_chains.public_types import DeployedServiceDescriptor as B

    assert A is B


# ---- run_local: typed dependency exposes descriptor helpers ------------------


def test_chain_runs_locally_with_deployed_dep():
    """``run_local`` + ``chainlet_to_service`` surfaces descriptors on ``DeploymentContext``."""

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
    assert desc.predict_url == "https://chain-abc.api.baseten.co/.../echo/run_remote"
    assert desc.internal_url is not None
    assert (
        desc.internal_url.gateway_run_remote_url
        == "https://wp.api.baseten.co/.../echo/run_remote"
    )
    assert desc.internal_url.hostname == "chain-abc.api.baseten.co"


# ---- Plain Truss bring-your-own-client -------------------------------------


def test_plain_truss_picks_up_siblings(
    tmp_path, dynamic_config_mount_dir, plain_truss_model
):
    """Plain model fixture resolves sibling URLs from dynamic config."""
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

    m = plain_truss_model()
    m.load()
    out = m.predict({})
    assert out["diarizer_url"] == "https://wp.api.baseten.co/.../diarizer/run_remote"
    assert out["auth_headers"] == {
        "Authorization": "Api-Key <from-secrets>",
        "Host": "chain-abc.api.baseten.co",
    }


def test_plain_truss_runs_standalone(
    tmp_path, dynamic_config_mount_dir, plain_truss_model
):
    """Without dynamic config, ``load`` skips sibling wiring."""
    m = plain_truss_model()
    m.load()
    out = m.predict({})
    assert out["diarizer_url"] is None
    assert out["auth_headers"] is None
