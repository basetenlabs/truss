import logging
from types import SimpleNamespace

from truss_chains.deployment.deployment_client import (
    _get_chain_root,
    _log_chain_workspace,
    _log_generated_chainlets,
)


class _Entrypoint:
    pass


def test_chain_workspace_dir_is_logged_once(caplog):
    with caplog.at_level(logging.INFO):
        chain_root = _get_chain_root(_Entrypoint)
        _get_chain_root(_Entrypoint)
        _log_chain_workspace(chain_root)

    messages = [
        record.message
        for record in caplog.records
        if "Using chain workspace dir" in record.message
    ]
    assert messages == [
        f"Using chain workspace dir: `{chain_root}` (files under this dir will "
        "be included as dependencies in the remote deployments and are importable)."
    ]


def test_generated_chainlets_share_one_directory_log(caplog, tmp_path):
    chain_root = tmp_path / "src" / "orchestrator"
    gen_dir = tmp_path / ".chains_generated" / "Voice_Agent"
    generated = []
    for entity_type, name in (
        ("TRUSS_CHAINLET", "STT"),
        ("TRUSS_CHAINLET", "LLM"),
        ("CHAINLET", "Orchestrator"),
    ):
        descriptor = SimpleNamespace(
            name=name, chainlet_cls=SimpleNamespace(entity_type=entity_type)
        )
        generated.append((descriptor, gen_dir / f"chainlet_{name}"))

    with caplog.at_level(logging.INFO):
        _log_generated_chainlets(chain_root, generated)

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.count(str(chain_root)) == 1
    assert message.count(str(gen_dir)) == 1
    assert "chainlet_STT" not in message
    assert "TRUSS_CHAINLET `STT`" in message
    assert "TRUSS_CHAINLET `LLM`" in message
    assert "CHAINLET `Orchestrator`" in message
