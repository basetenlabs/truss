"""Tests for ``chains.TrussChainlet`` — Project 2 of the Composable Chains
effort. Covers declaration validation, dep validation in another chainlet,
codegen branches, backward compatibility, and adversarial cases.

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


# ---- Declaration validation -------------------------------------------------


def test_declares_truss_dir_required():
    """Subclass without `truss_dir` should record a MISSING_API_ERROR."""

    class _Bad(chains.TrussChainlet):  # noqa: N801
        pass

    with pytest.raises(public_types.ChainsUsageError, match="must declare"):
        framework.raise_validation_errors()


def test_truss_dir_must_resolve_to_directory(tmp_path):
    bad_path = str(tmp_path / "does_not_exist")

    class _Bad(chains.TrussChainlet):
        truss_dir = bad_path

    with pytest.raises(public_types.ChainsUsageError, match="not a directory"):
        framework.raise_validation_errors()


def test_truss_dir_missing_config_yaml(tmp_path):
    bad = tmp_path / "bad_truss"
    bad.mkdir()
    bad_path = str(bad)

    class _Bad(chains.TrussChainlet):
        truss_dir = bad_path

    with pytest.raises(public_types.ChainsUsageError, match="missing `config.yaml`"):
        framework.raise_validation_errors()


def test_truss_dir_invalid_config_yaml(tmp_path):
    bad = tmp_path / "bad_truss"
    bad.mkdir()
    (bad / "config.yaml").write_text("not: a valid: truss config { broken")
    bad_path = str(bad)

    class _Bad(chains.TrussChainlet):
        truss_dir = bad_path

    with pytest.raises(public_types.ChainsUsageError, match="invalid `config.yaml`"):
        framework.raise_validation_errors()


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


# ---- Dep validation in another chainlet -------------------------------------


def test_chainletbase_can_depend_on_truss_chainlet(truss_dir_path):
    """A typed ChainletBase can declare a TrussChainlet as a dep with the
    DeployedServiceDescriptor type annotation."""
    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: chains.DeployedServiceDescriptor = chains.depends(_Echo),
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
            echo: chains.DeployedServiceDescriptor = chains.depends(_Echo),
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
            echo: chains.DeployedServiceDescriptor = chains.depends(_Echo),
            context: chains.DeploymentContext = chains.depends_context(),
        ):
            pass

    framework.raise_validation_errors()
    caller_desc = framework.get_descriptor(_Caller)
    assert "echo" in caller_desc.dependencies
    dep = caller_desc.dependencies["echo"]
    assert dep.chainlet_cls is _Echo
    assert framework.is_truss_chainlet(dep.chainlet_cls)


# ---- Codegen branches -------------------------------------------------------


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

    # User's other config keys preserved.
    assert dst_config["model_name"] == "EchoTruss"
    assert dst_config["python_version"] == "py311"


def test_entrypoint_codegen_emits_descriptor_injection_for_truss_dep(
    truss_dir_path, tmp_path
):
    """The entrypoint's generated `model.py` must call
    `self._context.get_service_descriptor(...)` for TrussChainlet deps,
    not `stub.factory(...)`."""
    from truss_chains.deployment import code_gen

    truss_dir_str = str(truss_dir_path)

    class _Echo(chains.TrussChainlet):
        truss_dir = truss_dir_str

    class _Caller(chains.ChainletBase):
        async def run_remote(self, x: str) -> str:
            return x

        def __init__(
            self,
            echo: chains.DeployedServiceDescriptor = chains.depends(_Echo),
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

    # Descriptor injection used.
    assert (
        "get_service_descriptor('_Echo')" in generated_model_py
        or 'get_service_descriptor("_Echo")' in generated_model_py
    )
    # No stub.factory(_Echo, ...) — no typed stub for TrussChainlet deps.
    assert "stub.factory(_Echo" not in generated_model_py
    # No `class _Echo(stub.StubBase)` — no generated stub class.
    assert "class _Echo(stub" not in generated_model_py


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
