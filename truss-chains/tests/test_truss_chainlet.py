"""Tests for ``chains.TrussChainlet``.

Sections, in order of the lifecycle they exercise:
  - TrussChainlet declaration (parse-time validation of the class body)
  - ChainletBase depending on TrussChainlet (dep validation)
  - TrussChainlet artifact generation (what ``_prepare_truss_chainlet_artifact``
    writes — the TC's own ``config.yaml`` + bundled user files)
  - Caller-side codegen (what the calling ChainletBase's generated ``model.py``
    looks like when it has TC deps)
  - TrussChainlet as chain entrypoint (``deps`` ClassVar + codegen)
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
    """An unparseable `config.yaml` is OK at parse time; the error surfaces
    at codegen."""
    from truss_chains.deployment import code_gen

    bad = tmp_path / "bad_truss"
    bad.mkdir()
    (bad / "config.yaml").write_text("not: a valid: truss config { broken")
    bad_path = str(bad)

    class _Bad(chains.TrussChainlet):
        truss_dir = bad_path

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Bad)

    with pytest.raises(public_types.ChainsUsageError, match="invalid `config.yaml`"):
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
        assert _Echo._resolved_truss_dir == rel_truss_dir.resolve()
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
    ``chains.ServiceHandle`` type annotation."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: chains.ServiceHandle = chains.depends(_Echo),
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
            echo: chains.ServiceHandle = chains.depends(_Echo),
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
            echo: chains.ServiceHandle = chains.depends(_Echo),
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
    ``chains.ServiceHandle``. The legacy ``chains.DeployedServiceDescriptor``
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


def test_entrypoint_codegen_emits_service_handle_for_truss_dep(
    truss_dir_path, tmp_path
):
    """The entrypoint's generated ``model.py`` must construct a
    :class:`truss_chains.ServiceHandle` for each TrussChainlet dep, not call
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
            echo: chains.ServiceHandle = chains.depends(_Echo),
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

    # ServiceHandle injection used (display_name-keyed).
    assert (
        "ServiceHandle('_Echo')" in generated_model_py
        or 'ServiceHandle("_Echo")' in generated_model_py
    )
    assert "from truss_chains import ServiceHandle" in generated_model_py
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
            echo: chains.ServiceHandle = chains.depends(
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
    """A CB with multiple TC deps gets one ``ServiceHandle(...)`` injection
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
            echo1: chains.ServiceHandle = chains.depends(_Echo1),
            echo2: chains.ServiceHandle = chains.depends(_Echo2),
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

    # Both ServiceHandle injections emitted.
    assert (
        "ServiceHandle('_Echo1')" in generated or 'ServiceHandle("_Echo1")' in generated
    )
    assert (
        "ServiceHandle('_Echo2')" in generated or 'ServiceHandle("_Echo2")' in generated
    )
    # Import emitted once (set semantics — not duplicated).
    assert generated.count("from truss_chains import ServiceHandle") == 1


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
            echo: chains.ServiceHandle = chains.depends(_Echo),
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
            echo: chains.ServiceHandle = chains.depends(_Echo),
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
    byte-for-byte (model.py preserved exactly), and merge `chains_metadata`
    into the copied `config.yaml`."""
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

    # Generated config.yaml has the chains_metadata block.
    dst_config = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    chains_meta = dst_config["model_metadata"][private_types.TRUSS_CONFIG_CHAINS_KEY]
    assert "chainlet_to_service" in chains_meta
    # Empty for a leaf TrussChainlet (no nested deps).
    assert chains_meta["chainlet_to_service"] == {}

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


def test_truss_chainlet_artifact_auto_adds_chain_api_key(truss_dir_path, tmp_path):
    """The codegen artifact must include `baseten_chain_api_key` in its
    `secrets` map so any TrussChainlet that calls a sibling via the chain
    RPC URL has the chain-internal key available (Issue 2). The ChainletBase
    path auto-adds this; the TrussChainlet path must too."""
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
    assert public_types.CHAIN_API_KEY_SECRET_NAME in dst_config["secrets"]
    assert (
        dst_config["secrets"][public_types.CHAIN_API_KEY_SECRET_NAME]
        == public_types.SECRET_DUMMY
    )


def test_truss_chainlet_artifact_chain_api_key_idempotent(tmp_path):
    """If the user's truss `config.yaml` already lists `baseten_chain_api_key`
    in its `secrets` map, the codegen must not overwrite it — same idempotency
    contract as the ChainletBase path."""
    from truss_chains.deployment import code_gen

    # User truss with chain_api_key already declared.
    user_config = (
        VALID_TRUSS_CONFIG_YAML
        + f"secrets:\n  {public_types.CHAIN_API_KEY_SECRET_NAME}: null\n"
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
        chain_root=tmp_path,
        chain_name="codegen-test",
        chainlet_descriptor=descriptor,
        model_name="_Echo-abc12345",
    )

    dst_config = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    # User's literal value (None) preserved — codegen did NOT overwrite to '***'.
    assert (
        dst_config["secrets"][public_types.CHAIN_API_KEY_SECRET_NAME]
        != public_types.SECRET_DUMMY
    )


def test_truss_chainlet_artifact_preserves_user_model_metadata(tmp_path):
    """User-set ``model_metadata`` keys in the source ``config.yaml`` are
    preserved when the artifact merges ``chains_metadata`` — codegen must not
    clobber unrelated metadata."""
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
    # chains_metadata block added.
    assert private_types.TRUSS_CONFIG_CHAINS_KEY in dst_config["model_metadata"]
    # User-set sibling keys preserved.
    assert dst_config["model_metadata"]["custom_key"] == "foo"
    assert dst_config["model_metadata"]["nested"] == {"a": 1, "b": 2}


# ---- TrussChainlet as chain entrypoint --------------------------------------


def test_truss_chainlet_can_be_entrypoint(truss_dir_path):
    """A `TrussChainlet` decorated with `@chains.mark_entrypoint` registers
    as a valid entrypoint candidate. The `ChainletImporter._is_target_cls`
    override is what unlocks this; without it the importer skips
    TrussChainlets at discovery time."""
    truss_dir_str = str(truss_dir_path)

    @chains.mark_entrypoint("Polyglot Front Door")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Entry)
    assert descriptor.chainlet_cls.meta_data.is_entrypoint is True
    assert descriptor.chainlet_cls.meta_data.chain_name == "Polyglot Front Door"
    assert descriptor.is_truss_chainlet is True
    # `ChainletImporter._is_target_cls` accepts this class.
    assert framework.ChainletImporter._is_target_cls(_Entry)


def test_truss_chainlet_deps_class_attr_parses(truss_dir_path):
    """A `TrussChainlet` with `deps = [SomeDep]` populates the
    `dependencies` map on its descriptor — same shape ChainletBase produces
    from `__init__` deps."""
    truss_dir_str = str(truss_dir_path)

    class _Sibling(chains.ChainletBase):
        remote_config = chains.RemoteConfig(
            compute=chains.Compute(cpu_count=1, memory="512Mi")
        )

        async def run_remote(self, text: str) -> str:
            return text

    @chains.mark_entrypoint("Entry")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str
        deps = [_Sibling]

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Entry)
    assert "_Sibling" in descriptor.dependencies
    assert descriptor.dependencies["_Sibling"].chainlet_cls is _Sibling


def test_truss_chainlet_entrypoint_with_mixed_deps(truss_dir_path, tmp_path):
    """Entrypoint TrussChainlet can declare deps that are a mix of
    `ChainletBase` and other `TrussChainlet` types — both shapes are valid
    sibling targets."""
    cb_sibling_truss_dir = str(truss_dir_path)
    tc_sibling_truss_dir = tmp_path / "tc_sibling"
    tc_sibling_truss_dir.mkdir()
    (tc_sibling_truss_dir / "config.yaml").write_text(VALID_TRUSS_CONFIG_YAML)
    (tc_sibling_truss_dir / "model").mkdir()
    (tc_sibling_truss_dir / "model" / "model.py").write_text(VALID_MODEL_PY)

    class _CBSibling(chains.ChainletBase):
        remote_config = chains.RemoteConfig(
            compute=chains.Compute(cpu_count=1, memory="512Mi")
        )

        async def run_remote(self, text: str) -> str:
            return text

    class _TCSibling(chains.TrussChainlet):
        truss_dir = str(tc_sibling_truss_dir)

    @chains.mark_entrypoint("Entry")
    class _Entry(chains.TrussChainlet):
        truss_dir = cb_sibling_truss_dir
        deps = [_CBSibling, _TCSibling]

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Entry)
    assert set(descriptor.dependencies.keys()) == {"_CBSibling", "_TCSibling"}
    assert descriptor.dependencies["_CBSibling"].chainlet_cls is _CBSibling
    assert descriptor.dependencies["_TCSibling"].chainlet_cls is _TCSibling


def test_truss_chainlet_entrypoint_duplicate_dep_rejected(truss_dir_path):
    truss_dir_str = str(truss_dir_path)

    class _Sibling(chains.ChainletBase):
        remote_config = chains.RemoteConfig(
            compute=chains.Compute(cpu_count=1, memory="512Mi")
        )

        async def run_remote(self, text: str) -> str:
            return text

    @chains.mark_entrypoint("Entry")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str
        deps = [_Sibling, _Sibling]

    with pytest.raises(public_types.ChainsUsageError, match="duplicate"):
        framework.raise_validation_errors()


def test_truss_chainlet_entrypoint_self_dep_rejected(truss_dir_path):
    truss_dir_str = str(truss_dir_path)

    # Self-reference via a forward declaration: a TrussChainlet that lists
    # itself as a dep. We construct this by patching `deps` post-class.
    @chains.mark_entrypoint("Entry")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str

    # Clear what the class declaration registered (since `deps = []` passed),
    # then re-validate with self-reference.
    framework._global_error_collector.clear()
    framework._global_chainlet_registry.unregister_chainlet("_Entry")
    _Entry.deps = [_Entry]
    framework.validate_and_register_cls(_Entry)
    with pytest.raises(public_types.ChainsUsageError, match="cannot reference itself"):
        framework.raise_validation_errors()


def test_truss_chainlet_entrypoint_non_chainlet_dep_rejected(truss_dir_path):
    """A non-chainlet class in `deps` is rejected with a clear error."""
    truss_dir_str = str(truss_dir_path)

    class _NotAChainlet:
        pass

    @chains.mark_entrypoint("Entry")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str
        deps = [_NotAChainlet]

    with pytest.raises(
        public_types.ChainsUsageError, match="not a `ChainletBase` or `TrussChainlet`"
    ):
        framework.raise_validation_errors()


def test_truss_chainlet_entrypoint_deps_must_be_list(truss_dir_path):
    """`deps` must be a list or tuple — anything else is a clear type error."""
    truss_dir_str = str(truss_dir_path)

    @chains.mark_entrypoint("Entry")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str
        deps = "not a list"  # type: ignore[assignment]

    with pytest.raises(public_types.ChainsUsageError, match="must be a list"):
        framework.raise_validation_errors()


def test_truss_chainlet_entrypoint_codegen_preserves_truss_dir(
    truss_dir_path, tmp_path
):
    """When a TrussChainlet *is* the entrypoint, codegen still uses the
    `_prepare_truss_chainlet_artifact` path (not the framework-generated
    `model.py` path). The user's `truss_dir` is copied byte-for-byte, with
    only `model_metadata.chains_metadata` injected."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    @chains.mark_entrypoint("Polyglot")
    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Entry)
    chainlet_dir = code_gen.gen_truss_chainlet(
        chain_root=tmp_path,
        chain_name="codegen-test",
        chainlet_descriptor=descriptor,
        model_name="_Entry-abc12345",
    )

    # User's model.py is preserved.
    assert (chainlet_dir / "model" / "model.py").read_text() == VALID_MODEL_PY
    # config.yaml carries the uniquified model_name + chains_metadata.
    dst_config = yaml.safe_load((chainlet_dir / "config.yaml").read_text())
    assert dst_config["model_name"] == "_Entry-abc12345"
    assert private_types.TRUSS_CONFIG_CHAINS_KEY in (
        dst_config.get("model_metadata") or {}
    )


def test_truss_chainlet_entrypoint_with_truss_chainlet_dep(truss_dir_path):
    """A ``TrussChainlet`` entrypoint can depend on another ``TrussChainlet``
    via the ``deps`` ClassVar. The dep is recorded on the descriptor and
    codegen treats it as a TC dep (no typed stub generated)."""
    truss_dir_str = str(truss_dir_path)

    class _Inner(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Entry(chains.TrussChainlet):
        truss_dir = truss_dir_str
        deps = [_Inner]

    framework.raise_validation_errors()
    entry_desc = framework.get_descriptor(_Entry)
    assert "_Inner" in entry_desc.dependencies
    dep = entry_desc.dependencies["_Inner"]
    assert framework.is_truss_chainlet(dep.chainlet_cls)
    assert dep.chainlet_cls is _Inner


def test_truss_chainlet_entrypoint_with_empty_deps(truss_dir_path):
    """A ``TrussChainlet`` entrypoint with ``deps = []`` is a standalone
    polyglot front door — no siblings, just an externally-callable Truss."""
    truss_dir_str = str(truss_dir_path)

    class _Solo(chains.TrussChainlet):
        truss_dir = truss_dir_str
        deps = []

    framework.raise_validation_errors()
    descriptor = framework.get_descriptor(_Solo)
    assert descriptor.dependencies == {}


def test_chainletbase_ws_as_sibling_rejected():
    """WebSocket `ChainletBase` chainlets can only be entrypoints, not
    siblings. The validator at `framework.py:_validate_dependencies` checks
    `endpoint.is_websocket` on each dep. Locks in that behavior for the
    Project 3 work — `TrussChainlet` entrypoints with a `ChainletBase`+WS
    dep should also be blocked via the same predicate (we route
    `_validate_truss_chainlet_deps` through the same check)."""

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
