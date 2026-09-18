"""Tests for ``chains.TrussChainlet``.

Sections, in order of the lifecycle they exercise:
  - TrussChainlet declaration (parse-time validation of the class body)
  - ChainletBase depending on TrussChainlet (dep validation)
  - TrussChainlet artifact generation (what ``_prepare_truss_chainlet_artifact``
    writes — the TC's own ``config.yaml`` + bundled user files)
  - Caller-side codegen (what the calling ChainletBase's generated ``model.py``
    looks like when it has TC deps)
  - Leaf-only contract (TrussChainlets rejected as entrypoints; ``deps``
    ClassVar silently ignored)
  - run_local with TrussChainlet deps
  - Backward compatibility (pure ChainletBase chains unaffected)
  - Adversarial (bad inputs rejected)

The validation layer collects errors rather than raising them — tests use
``framework.raise_validation_errors`` to surface them as a single
``ChainsUsageError``.

Note on fixture naming: the test fixture is `truss_dir_path` (not `truss_dir`)
to avoid shadowing the `truss_dir` class attribute that `TrussChainlet`
subclasses must declare. Pytest fixture name resolution interacts oddly with
class-body assignments using the same name.
"""

import pathlib

import pytest
import yaml

import truss_chains as chains
from truss_chains import framework, private_types, public_types
from truss_chains.remote_chainlet.truss_chainlet import TrussHandle

# ---- Fixtures ---------------------------------------------------------------


VALID_TRUSS_CONFIG_YAML = """\
model_name: EchoTruss
model_class_filename: model.py
model_class_name: Model
python_version: py311
resources:
  cpu: '1'
  memory: 512Mi
  use_gpu: false
"""


VALID_MODEL_PY = """\
class Model:
    def __init__(self, **kwargs):
        pass

    def predict(self, request):
        return {"out": str(request.get("text", "")).upper()}
"""


@pytest.fixture
def truss_dir_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal valid Truss directory."""
    d = tmp_path / "echo_truss"
    d.mkdir()
    (d / "config.yaml").write_text(VALID_TRUSS_CONFIG_YAML)
    (d / "model").mkdir()
    (d / "model" / "model.py").write_text(VALID_MODEL_PY)
    return d


@pytest.fixture(autouse=True)
def reset_framework_state():
    """Each test starts with a clean error collector and a snapshot-restored
    chainlet registry — test-local classes share names like `_Echo`, which
    would otherwise collide on the global registry across tests. We save and
    restore the registry contents instead of clearing wholesale, so chainlets
    registered at module-import time by *other* test files remain available."""
    framework._global_error_collector.clear()
    snapshot_chainlets = dict(framework._global_chainlet_registry._chainlets)
    snapshot_names = dict(framework._global_chainlet_registry._name_to_cls)
    framework._global_chainlet_registry._chainlets.clear()
    framework._global_chainlet_registry._name_to_cls.clear()
    try:
        yield
    finally:
        framework._global_error_collector.clear()
        framework._global_chainlet_registry._chainlets.clear()
        framework._global_chainlet_registry._chainlets.update(snapshot_chainlets)
        framework._global_chainlet_registry._name_to_cls.clear()
        framework._global_chainlet_registry._name_to_cls.update(snapshot_names)


# ---- TrussChainlet declaration ----------------------------------------------


def test_declares_truss_dir_required():
    """Subclass without `truss_dir` should record a MISSING_API_ERROR."""

    class _Bad(chains.TrussChainlet):  # noqa: N801
        pass

    with pytest.raises(public_types.ChainsUsageError, match="must declare"):
        framework.raise_validation_errors()


def test_truss_dir_must_resolve_to_directory(tmp_path):
    """A missing `truss_dir` is OK at parse time (matches ChainletBase, which
    has no filesystem-state checks at parse). The error surfaces at codegen
    time only for chainlets that are actually being built/deployed — which
    keeps `truss chains watch --experimental-chainlet-names <subset>` working
    when non-targeted chainlets' truss_dirs are absent locally."""
    from truss_chains.deployment import code_gen

    bad_path = str(tmp_path / "does_not_exist")

    class _Bad(chains.TrussChainlet):
        truss_dir = bad_path

    # Parse-time validation now succeeds (source-only invariants pass).
    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Bad)

    # Codegen surfaces the error at the natural phase.
    with pytest.raises(public_types.ChainsUsageError, match="not a directory"):
        code_gen.gen_truss_chainlet(
            chain_root=tmp_path,
            chain_name="codegen-test",
            chainlet_descriptor=descriptor,
        )


def test_truss_dir_missing_config_yaml(tmp_path):
    """An existing directory without a `config.yaml` is OK at parse time; the
    error surfaces at codegen."""
    from truss_chains.deployment import code_gen

    bad = tmp_path / "bad_truss"
    bad.mkdir()
    bad_path = str(bad)

    class _Bad(chains.TrussChainlet):
        truss_dir = bad_path

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Bad)

    with pytest.raises(public_types.ChainsUsageError, match="missing `config.yaml`"):
        code_gen.gen_truss_chainlet(
            chain_root=tmp_path,
            chain_name="codegen-test",
            chainlet_descriptor=descriptor,
        )


def test_truss_dir_invalid_config_yaml(tmp_path):
    """An unparseable `config.yaml` is OK at parse time; the underlying yaml /
    pydantic exception bubbles unchanged at codegen (no framework-level wrap)."""
    from truss_chains.deployment import code_gen

    bad = tmp_path / "bad_truss"
    bad.mkdir()
    (bad / "config.yaml").write_text("not: a valid: truss config { broken")
    bad_path = str(bad)

    class _Bad(chains.TrussChainlet):
        truss_dir = bad_path

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Bad)

    with pytest.raises(yaml.YAMLError):
        code_gen.gen_truss_chainlet(
            chain_root=tmp_path,
            chain_name="codegen-test",
            chainlet_descriptor=descriptor,
        )


def test_truss_dir_resolved_relative_to_declaring_file():
    """Relative `truss_dir` should resolve relative to the file declaring
    the subclass — not the cwd."""
    test_dir = pathlib.Path(__file__).parent
    rel_truss_dir = test_dir / "_relative_test_truss"
    rel_truss_dir.mkdir(exist_ok=True)
    try:
        (rel_truss_dir / "config.yaml").write_text(VALID_TRUSS_CONFIG_YAML)
        (rel_truss_dir / "model").mkdir(exist_ok=True)
        (rel_truss_dir / "model" / "model.py").write_text(VALID_MODEL_PY)

        class _Echo(chains.TrussChainlet):
            truss_dir = "_relative_test_truss"  # relative to this test file

        framework.raise_validation_errors()  # should not raise
        descriptor = framework.get_descriptor(_Echo)
        assert descriptor.truss_dir == rel_truss_dir.resolve()
    finally:
        import shutil

        shutil.rmtree(rel_truss_dir, ignore_errors=True)


def test_truss_dir_wrong_type_rejected():
    """``truss_dir`` must be ``str`` or ``pathlib.Path``; other types are rejected
    at parse time with a distinct error from the missing-attribute case."""

    class _Bad(chains.TrussChainlet):
        truss_dir = 123  # type: ignore[assignment]

    with pytest.raises(
        public_types.ChainsUsageError, match="must be a str or pathlib.Path"
    ):
        framework.raise_validation_errors()


def test_valid_truss_chainlet_registers_descriptor(truss_dir_path):
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Echo)
    assert descriptor.is_truss_chainlet
    assert descriptor.chainlet_cls is _Echo
    assert descriptor.truss_dir == truss_dir_path
    # Empty deps + dummy endpoint.
    assert descriptor.dependencies == {}
    assert descriptor.endpoint == framework._DUMMY_ENDPOINT_DESCRIPTOR


# ---- ChainletBase depending on TrussChainlet --------------------------------


def test_chainletbase_can_depend_on_truss_chainlet(truss_dir_path):
    """A typed ChainletBase can declare a TrussChainlet as a dep with the
    ``TrussHandle`` type annotation."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: TrussHandle = chains.depends(_Echo),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            self._echo = echo

    framework.raise_validation_errors()


def test_mixed_deps_typed_and_truss(truss_dir_path):
    """ChainletBase entrypoint with both a typed ChainletBase dep and a
    TrussChainlet dep — both validate."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Reverser(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x[::-1]

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            reverser: _Reverser = chains.depends(_Reverser),
            echo: TrussHandle = chains.depends(_Echo),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            self._reverser = reverser
            self._echo = echo

    framework.raise_validation_errors()


def test_truss_chainlet_dep_recorded_on_descriptor(truss_dir_path):
    """The dep entry on the entrypoint's descriptor records the TrussChainlet
    by class — codegen later branches on `is_truss_chainlet(dep.chainlet_cls)`."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: TrussHandle = chains.depends(_Echo),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            pass

    framework.raise_validation_errors()
    caller_desc = framework.get_descriptor(_Caller)
    assert "echo" in caller_desc.dependencies
    dep = caller_desc.dependencies["echo"]
    assert dep.chainlet_cls is _Echo
    assert framework.is_truss_chainlet(dep.chainlet_cls)


def test_truss_chainlet_dep_with_descriptor_annotation_rejected(truss_dir_path):
    """After change #1, ``chains.depends(TC)`` must be annotated as
    ``TrussHandle``. The legacy ``chains.DeployedServiceDescriptor``
    annotation is rejected by the validator — locks in the migration."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            # Intentionally mismatched annotation — the test asserts the
            # framework validator rejects this (mypy would too, hence ignore).
            echo: chains.DeployedServiceDescriptor = chains.depends(_Echo),  # type: ignore[assignment]
        ):
            pass

    with pytest.raises(public_types.ChainsUsageError, match="type annotation"):
        framework.raise_validation_errors()


# ---- Caller-side codegen (ChainletBase with TrussChainlet deps) -------------


def test_entrypoint_codegen_emits_truss_handle_for_truss_dep(truss_dir_path, tmp_path):
    """The entrypoint's generated ``model.py`` must construct a
    :class:`TrussHandle` for each TrussChainlet dep, not call
    ``stub.factory(...)``. The user's ``__init__`` receives the handle and
    uses ``handle.http_call_args(...)`` for BYOC sibling calls."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: TrussHandle = chains.depends(_Echo),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            self._echo = echo

    framework.raise_validation_errors()

    descriptor = framework.get_descriptor(_Caller)
    chain_root = pathlib.Path(__file__).parent
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=chain_root, chain_name="codegen-test", chainlet_descriptor=descriptor
    )
    generated_model_py = (chainlet_dir / "model" / "model.py").read_text()

    # TrussHandle injection used (display_name-keyed).
    assert (
        "truss_chainlet.TrussHandle('_Echo')" in generated_model_py
        or 'truss_chainlet.TrussHandle("_Echo")' in generated_model_py
    )
    assert (
        "from truss_chains.remote_chainlet import truss_chainlet" in generated_model_py
    )
    # No stale descriptor-injection pattern.
    assert "get_service_descriptor" not in generated_model_py
    # No stub.factory(_Echo, ...) — no typed stub for TrussChainlet deps.
    assert "stub.factory(_Echo" not in generated_model_py
    # No `class _Echo(stub.StubBase)` — no generated stub class.
    assert "class _Echo(stub" not in generated_model_py


def test_truss_chainlet_dep_options_propagated(truss_dir_path):
    """``chains.depends(MyTC, retries=5, ...)`` propagates ``RPCOptions`` into
    the caller's chains_metadata.chainlet_to_service entry, matching the
    CB-dep behavior."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: TrussHandle = chains.depends(
                _Echo, retries=5, timeout_sec=42.0, concurrency_limit=7
            ),
        ):
            self._echo = echo

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Caller)

    # Options round-tripped onto the dep descriptor.
    dep = descriptor.dependencies["echo"]
    assert dep.options.retries == 5
    assert dep.options.timeout_sec == 42.0
    assert dep.options.concurrency_limit == 7

    # Options end up in the generated chains_metadata too.
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=pathlib.Path(__file__).parent,
        chain_name="codegen-test",
        chainlet_descriptor=descriptor,
    )
    config_yaml = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    chains_meta = config_yaml["model_metadata"][private_types.TRUSS_CONFIG_CHAINS_KEY]
    echo_service = chains_meta["chainlet_to_service"]["_Echo"]
    assert echo_service["options"]["retries"] == 5
    assert echo_service["options"]["timeout_sec"] == 42.0
    assert echo_service["options"]["concurrency_limit"] == 7


def test_chainletbase_with_multiple_truss_chainlet_deps_codegen(
    truss_dir_path, tmp_path
):
    """A CB with multiple TC deps gets one ``TrussHandle(...)`` injection
    per TC dep in the generated ``load()``."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    class _Echo1(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Echo2(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo1: TrussHandle = chains.depends(_Echo1),
            echo2: TrussHandle = chains.depends(_Echo2),
        ):
            self._echo1 = echo1
            self._echo2 = echo2

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Caller)
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=pathlib.Path(__file__).parent,
        chain_name="codegen-test",
        chainlet_descriptor=descriptor,
    )
    generated = (chainlet_dir / "model" / "model.py").read_text()

    # Both TrussHandle injections emitted.
    assert (
        "truss_chainlet.TrussHandle('_Echo1')" in generated
        or 'truss_chainlet.TrussHandle("_Echo1")' in generated
    )
    assert (
        "truss_chainlet.TrussHandle('_Echo2')" in generated
        or 'truss_chainlet.TrussHandle("_Echo2")' in generated
    )
    # Import emitted once (set semantics — not duplicated).
    assert (
        generated.count("from truss_chains.remote_chainlet import truss_chainlet") == 1
    )


# ---- run_local with TrussChainlet deps --------------------------------------


def test_run_local_truss_chainlet_dep_raises_clear_error(truss_dir_path):
    """``run_local`` can't instantiate a TrussChainlet (TC wraps a deployable
    directory, not an in-process class). The framework should raise a clean
    ``ChainsUsageError`` pointing at the workaround, not a confusing
    ``AssertionError`` from inside the marker-replacement loop."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: TrussHandle = chains.depends(_Echo),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            self._echo = echo

    framework.raise_validation_errors()

    with chains.run_local(secrets={}, chainlet_to_service={}):
        with pytest.raises(
            public_types.ChainsUsageError, match="cannot instantiate TrussChainlet dep"
        ):
            _Caller()


def test_run_local_truss_chainlet_dep_can_be_pre_supplied(truss_dir_path):
    """The error suggests pre-supplying a stand-in via kwarg; verify that
    pre-supplied kwargs bypass the marker-replacement loop entirely."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: TrussHandle = chains.depends(_Echo),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            self._echo = echo

    framework.raise_validation_errors()

    # Pre-supply a sentinel for the TC dep; framework should accept it
    # verbatim (no validation of the runtime type).
    sentinel = object()
    with chains.run_local(secrets={}, chainlet_to_service={}):
        caller = _Caller(echo=sentinel)  # type: ignore[arg-type]
    assert caller._echo is sentinel


# ---- Backward compatibility -------------------------------------------------


def test_pure_chainletbase_chain_unaffected():
    """A chain made entirely of ChainletBase chainlets validates identically
    to before this change."""

    class _Inner(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

    class _Outer(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            inner: _Inner = chains.depends(_Inner),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            self._inner = inner

    framework.raise_validation_errors()  # no errors
    inner_desc = framework.get_descriptor(_Inner)
    outer_desc = framework.get_descriptor(_Outer)
    assert not inner_desc.is_truss_chainlet
    assert not outer_desc.is_truss_chainlet
    assert "inner" in outer_desc.dependencies
    assert not framework.is_truss_chainlet(
        outer_desc.dependencies["inner"].chainlet_cls
    )


# ---- Adversarial ------------------------------------------------------------


def test_depends_on_random_class_rejected():
    class _Random:
        pass

    class _Caller(chains.ChainletBase):
        async def run_remote(self) -> None:
            pass

        def __init__(self, x=chains.depends(_Random)):  # type: ignore[arg-type]
            pass

    with pytest.raises(public_types.ChainsUsageError):
        framework.raise_validation_errors()


# ---- TrussChainlet artifact generation --------------------------------------


def test_truss_chainlet_artifact_preserves_user_files(truss_dir_path, tmp_path):
    """Generating a TrussChainlet artifact must copy the user's truss_dir
    byte-for-byte (model.py preserved exactly). Since TrussChainlets are
    non-entry leaves, the codegen also does NOT inject any chains_metadata."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Echo)
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=tmp_path, chain_name="codegen-test", chainlet_descriptor=descriptor
    )

    # User's model.py is preserved byte-for-byte.
    src_model_py = (truss_dir_path / "model" / "model.py").read_text()
    dst_model_py = (chainlet_dir / "model" / "model.py").read_text()
    assert src_model_py == dst_model_py

    dst_config = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    # No chains_metadata injection for leaf TCs.
    assert private_types.TRUSS_CONFIG_CHAINS_KEY not in (
        dst_config.get("model_metadata") or {}
    )

    # `model_name` is overwritten with the chain-uniquified name; user's
    # literal `EchoTruss` is preserved in the source `truss_dir` but not in
    # the generated artifact. Other user-set config keys remain preserved.
    assert dst_config["model_name"] != "EchoTruss"
    assert dst_config["python_version"] == "py311"


def test_truss_chainlet_artifact_overrides_user_model_name(truss_dir_path, tmp_path):
    """The codegen artifact must carry the chain-uniquified model_name, not
    the user's literal `truss_dir/config.yaml.model_name`. This avoids
    workspace name collisions on `chains push` (Issue 1) and `OracleVersion.name`
    regex failures during UI promotion (Issue 5)."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Echo)
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=tmp_path,
        chain_name="codegen-test",
        chainlet_descriptor=descriptor,
        model_name="_Echo-abc12345",
    )

    dst_config = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    assert dst_config["model_name"] == "_Echo-abc12345"

    # User's source file is untouched.
    src_config = yaml.safe_load((truss_dir_path / "config.yaml").read_text())
    assert src_config["model_name"] == "EchoTruss"


def test_truss_chainlet_artifact_does_not_inject_chain_api_key(
    truss_dir_path, tmp_path
):
    """TrussChainlets are non-entry leaves and make no outbound sibling
    calls, so the codegen must NOT auto-inject ``baseten_chain_api_key``
    into the user's ``config.yaml`` secrets map."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Echo)
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=tmp_path,
        chain_name="codegen-test",
        chainlet_descriptor=descriptor,
        model_name="_Echo-abc12345",
    )

    dst_config = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    secrets = dst_config.get("secrets") or {}
    assert public_types.CHAIN_API_KEY_SECRET_NAME not in secrets


def test_truss_chainlet_artifact_preserves_user_model_metadata(tmp_path):
    """User-set ``model_metadata`` keys in the source ``config.yaml`` are
    preserved verbatim. Since TrussChainlets have no deps, the codegen also
    does NOT inject any ``chains_metadata`` block."""
    from truss_chains.deployment import code_gen

    user_config = (
        VALID_TRUSS_CONFIG_YAML
        + "model_metadata:\n  custom_key: foo\n  nested:\n    a: 1\n    b: 2\n"
    )
    truss_dir = tmp_path / "user_truss"
    truss_dir.mkdir()
    (truss_dir / "config.yaml").write_text(user_config)
    (truss_dir / "model").mkdir()
    (truss_dir / "model" / "model.py").write_text(VALID_MODEL_PY)
    truss_dir_str = str(truss_dir)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Echo)
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=tmp_path, chain_name="codegen-test", chainlet_descriptor=descriptor
    )

    dst_config = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    metadata = dst_config.get("model_metadata") or {}
    # No chains_metadata injection for leaf TCs.
    assert private_types.TRUSS_CONFIG_CHAINS_KEY not in metadata
    # User-set keys preserved verbatim.
    assert metadata["custom_key"] == "foo"
    assert metadata["nested"] == {"a": 1, "b": 2}


# ---- Leaf-only contract: TrussChainlet rejected as entrypoint / deps unsupported


def test_truss_chainlet_rejected_as_entrypoint(truss_dir_path):
    """`TrussChainlet` is a non-entry leaf — `@chains.mark_entrypoint` on
    one must raise a clear ``ChainsUsageError`` so users don't get a confusing
    "no entrypoint found" failure from the importer."""
    truss_dir_str = str(truss_dir_path)

    @chains.mark_entrypoint("Polyglot Front Door")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str

    with pytest.raises(
        public_types.ChainsUsageError,
        match="TrussChainlet cannot be a chain entrypoint",
    ):
        framework.raise_validation_errors()


def test_truss_chainlet_deps_classvar_ignored(truss_dir_path):
    """A ``deps`` ClassVar on a ``TrussChainlet`` is silently ignored —
    TrussChainlets are leaves and have no framework-level deps mechanism.
    The descriptor's ``dependencies`` map is always empty."""
    truss_dir_str = str(truss_dir_path)

    class _Sibling(chains.ChainletBase):
        remote_config = chains.RemoteConfig(
            compute=chains.Compute(cpu_count=1, memory="512Mi")
        )

        async def run_remote(self, text: str) -> str:
            return text

    class _LeafWithDepsAttr(chains.TrussChainlet):
        truss_dir = truss_dir_str
        deps = [_Sibling]  # type: ignore[attr-defined]  # ignored by the framework

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_LeafWithDepsAttr)
    assert descriptor.dependencies == {}


def test_chainletbase_ws_as_sibling_rejected():
    """WebSocket `ChainletBase` chainlets can only be entrypoints, not
    siblings. The validator at `framework.py:_validate_dependencies` checks
    `endpoint.is_websocket` on each dep. Locks in that behavior — unrelated
    to the TrussChainlet rollback but kept here because it shares the
    sibling-dep validation surface."""

    class _WSChainlet(chains.ChainletBase):
        remote_config = chains.RemoteConfig(
            compute=chains.Compute(cpu_count=1, memory="512Mi")
        )

        async def run_remote(self, websocket: public_types.WebSocketProtocol) -> None:
            pass

    class _CallerCB(chains.ChainletBase):
        remote_config = chains.RemoteConfig(
            compute=chains.Compute(cpu_count=1, memory="512Mi")
        )

        def __init__(self, ws: _WSChainlet = chains.depends(_WSChainlet)) -> None:
            self._ws = ws

        async def run_remote(self, text: str) -> str:
            return text

    with pytest.raises(
        public_types.ChainsUsageError,
        match="websockets can only be used in the entrypoint",
    ):
        framework.raise_validation_errors()
